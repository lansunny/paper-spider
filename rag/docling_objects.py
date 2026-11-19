"""Build retrievable views and a source graph from an immutable Docling document."""
from collections import Counter, defaultdict
from hashlib import sha256
from html import escape
import re


LABEL = r'((?:[A-Z]\.?)?\d+(?:\.\d+)*[A-Z]?)(?!\w)'
NUMBER = re.compile(r'\b(fig(?:ures?|s)?\.?|tab(?:les?|s)?\.?)\s*(?:[:.\s]\s*)?' + LABEL, re.I)
MORE_NUMBERS = re.compile(r'\s*(?:,\s*(?:and\s+)?|\band\b\s*|&\s*)\s*' + LABEL, re.I)
SKIP = {'title', 'section_header', 'caption', 'page_header', 'page_footer', 'document_index'}
RENDER_SCALE = 2


def mentions(text):
    for match in NUMBER.finditer(text):
        kind = 'figure' if match[1].lower().startswith('fig') else 'table'
        yield kind, match[2].upper(), match.start(), match.end()
        end = match.end()
        while more := MORE_NUMBERS.match(text, end):
            yield kind, more[1].upper(), more.start(1), more.end(1)
            end = more.end()


def target_for_number(number, exact_ids):
    """A subfigure such as 2a can point at the whole Figure 2 if only it exists."""
    ids = exact_ids(number)
    if not ids and re.fullmatch(r'(?:[A-Z]\.?)?\d+(?:\.\d+)*[A-Z]', number):
        ids = exact_ids(number[:-1])
    return ids


def identity(*parts):
    return sha256('\x00'.join(parts).encode('utf-8')).hexdigest()[:32]


def windows(text, tokenizer, budget, overlap, prefix=''):
    """Keep semantic element boundaries; cut overlong elements into exact overlapping slices."""
    def count(body):
        return len(tokenizer.encode(prefix + body, add_special_tokens=True, truncation=False))

    if count(text) <= budget:
        yield 0, len(text)
        return
    offsets = [(a, b) for a, b in tokenizer(text, add_special_tokens=False, truncation=False,
                                           return_offsets_mapping=True)['offset_mapping'] if b > a]
    if not offsets:
        raise ValueError('Cannot map overlong text to tokenizer offsets')
    size = max(1, budget - count(''))
    start = 0
    while start < len(offsets):
        stop = min(start + size, len(offsets))
        left = 0 if start == 0 else offsets[start][0]
        while True:
            right = len(text) if stop == len(offsets) else max(b for _, b in offsets[start:stop])
            if count(text[left:right]) <= budget:
                break
            stop -= 1
            if stop <= start:
                raise ValueError('Embedding budget cannot fit a token with its metadata')
        yield left, right
        if stop == len(offsets):
            break
        start = max(start + 1, stop - overlap)


def table_views(data):
    """Return checked row views and HTML, retaining the complete raw cells elsewhere."""
    cells = data.get('table_cells', [])
    nrows, ncols = data.get('num_rows', 0), data.get('num_cols', 0)
    if not cells or nrows < 1 or ncols < 1 or nrows * ncols > 20000:
        return None, [], []
    grid = {}
    for i, cell in enumerate(cells):
        a, b, c, d = (cell[key] for key in ('start_row_offset_idx', 'end_row_offset_idx',
                                              'start_col_offset_idx', 'end_col_offset_idx'))
        if not (0 <= a < b <= nrows and 0 <= c < d <= ncols and
                b-a == cell['row_span'] and d-c == cell['col_span']):
            return None, [], []
        for r in range(a, b):
            for col in range(c, d):
                if (r, col) in grid:
                    return None, [], []
                grid[r, col] = i
    headers = defaultdict(list)
    for i, cell in enumerate(cells):
        if cell.get('column_header') and cell['text'].strip():
            for col in range(cell['start_col_offset_idx'], cell['end_col_offset_idx']):
                headers[col].append(i)
    for col in headers:
        headers[col].sort(key=lambda i: cells[i]['start_row_offset_idx'])
    html, rows = ['<table>'], []
    for r in range(nrows):
        html.append('<tr>')
        row_ids = list(dict.fromkeys(grid[r, c] for c in range(ncols) if (r, c) in grid))
        row_label = []
        for c in range(ncols):
            if (r, c) not in grid:
                html.append('<td></td>')
                continue
            cell = cells[grid[r, c]]
            if (r, c) != (cell['start_row_offset_idx'], cell['start_col_offset_idx']):
                continue
            tag = 'th' if any(cell.get(k) for k in ('column_header', 'row_header', 'row_section')) else 'td'
            html.append(f'<{tag} rowspan="{cell["row_span"]}" colspan="{cell["col_span"]}">{escape(cell["text"])}</{tag}>')
        html.append('</tr>')
        values = []
        for i in row_ids:
            cell = cells[i]
            text = cell['text'].strip()
            if not text or cell.get('column_header'):
                continue
            col = cell['start_col_offset_idx']
            if cell.get('row_header') or cell.get('row_section'):
                row_label.append(text)
                continue
            # A cell spanning unrelated columns cannot borrow the first column's title.
            paths = [tuple(j for j in headers[k] if cells[j]['end_row_offset_idx'] <= r)
                     for k in range(col, cell['end_col_offset_idx'])]
            path = paths[0] if all(p == paths[0] for p in paths) else ()
            names = ' / '.join(cells[j]['text'].strip() for j in path)
            values.append(f'{names}: {text}' if names else text)
        if values:
            rows.append({'row': r, 'cell_ids': row_ids, 'text': ' | '.join(row_label + values)})
    html.append('</table>')
    columns = list(dict.fromkeys(cells[i]['text'] for col in sorted(headers) for i in headers[col]))
    return ''.join(html), rows, columns


def crop(pdf, doc, prov, folder, object_id, i):
    import pymupdf

    n = prov['page_no']
    original = doc['pages'][str(n)]['size']
    box = prov['bbox']
    if box['coord_origin'] == 'BOTTOMLEFT':
        top, bottom = original['height'] - box['t'], original['height'] - box['b']
    elif box['coord_origin'] == 'TOPLEFT':
        top, bottom = box['t'], box['b']
    else:
        raise ValueError('Unknown Docling coordinate origin')
    page = pdf[n-1]
    sx, sy = page.rect.width / original['width'], page.rect.height / original['height']
    rect = pymupdf.Rect(box['l']*sx, top*sy, box['r']*sx, bottom*sy) & page.rect
    if rect.width < 1 or rect.height < 1:
        raise ValueError('Empty object crop')
    target = folder / 'crops' / f'{object_id}-{i}.png'
    target.parent.mkdir(exist_ok=True, parents=True)
    png = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE), clip=rect, alpha=False)
    png.save(str(target))
    return {'page': n, 'rect_pymupdf': list(rect), 'path': target.relative_to(folder).as_posix(),
            'width': png.width, 'height': png.height, 'sha256': sha256(target.read_bytes()).hexdigest()}, page.get_text('text', clip=rect).strip()


def materialize(doc, pdf, folder, pid, pdf_hash, parse_id, title, tokenizer, budget, overlap):
    nodes = {n['self_ref']: n for key in ('groups', 'texts', 'pictures', 'tables', 'form_items', 'key_value_items')
             for n in doc.get(key, [])}
    objects, edges, chunks, notices = [], [], [], Counter()
    by_number, heading_levels = defaultdict(list), {}

    def children(item):
        for ref in item.get('children', []):
            child = nodes[ref['$ref']]
            yield child
            yield from children(child)

    def refs(item, key):
        return [nodes[r['$ref']] for r in item.get(key, []) if r['$ref'] in nodes]

    def new(item, kind, headings, ordinal):
        obj = {'object_id': identity(pid, pdf_hash, parse_id, item['self_ref']), 'paper_id': pid, 'pdf_sha256': pdf_hash,
               'ref': item['self_ref'], 'kind': kind, 'label': item['label'], 'headings': list(headings),
               'ordinal': ordinal, 'provenance': item.get('prov', []), 'text': item.get('text', '')}
        objects.append(obj)
        return obj

    def current_title():
        return [heading_levels[level] for level in sorted(heading_levels)]

    def walk(item):
        for ref in item.get('children', []):
            child = nodes[ref['$ref']]
            if child['self_ref'].startswith('#/groups/'):
                yield from walk(child)
            else:
                yield child
                if child['label'] not in {'table', 'picture'}:
                    yield from walk(child)

    for ordinal, item in enumerate(walk(doc['body'])):
        label = item['label']
        if label == 'section_header':
            level = item.get('level', 1)
            heading_levels = {k: v for k, v in heading_levels.items() if k < level}
            heading_levels[level] = item.get('text', '')
        if label in {'picture', 'table'}:
            kind = 'figure' if label == 'picture' else 'table'
            obj = new(item, kind, current_title(), ordinal)
            captions, footnotes = refs(item, 'captions'), refs(item, 'footnotes')
            for key, values in [('captions', captions), ('footnotes', footnotes)]:
                obj[key] = [{'ref': n['self_ref'], 'text': n.get('text', ''), 'provenance': n.get('prov', [])} for n in values]
            obj['caption'] = '\n'.join(n.get('text', '') for n in captions if n.get('text', '').strip())
            obj['text'] = obj['caption']
            obj['crops'] = []
            native_text = []
            for i, prov in enumerate(obj['provenance']):
                try:
                    image, text = crop(pdf, doc, prov, folder, obj['object_id'], i)
                    obj['crops'].append(image)
                    if text:
                        native_text.append(text)
                except (ValueError, RuntimeError, ZeroDivisionError, KeyError, IndexError):
                    notices['invalid_crop_regions'] += 1
            children_text = [n.get('text', '') for n in children(item) if n['label'] not in {'caption', 'footnote'}]
            obj['region_text'] = '\n'.join(dict.fromkeys(t.strip() for t in children_text + native_text if t.strip()))
            numbers = set()
            for cap in captions:
                text = cap.get('text', '')
                match = NUMBER.match(text.lstrip())
                if match and next(mentions(match[0]))[0] == kind:
                    numbers.add(match[2].upper())
            obj['labels'] = sorted(numbers)
            if not numbers:
                notices['objects_without_number'] += 1
            for number in numbers:
                by_number[kind, number].append(obj['object_id'])
            if kind == 'table':
                obj['table'] = {k: v for k, v in item.get('data', {}).items() if k != 'grid'}
                obj['html'], obj['rows'], obj['columns'] = table_views(obj['table'])
                if obj['html'] is None:
                    notices['invalid_table_grid'] += 1
        elif label not in SKIP and item.get('text', '').strip():
            obj = new(item, 'paragraph', current_title(), ordinal)

    for obj in objects:
        if obj['kind'] != 'paragraph':
            continue
        for kind, number, left, right in mentions(obj['text']):
            ids = target_for_number(number, lambda n: by_number.get((kind, n), []))
            if len(ids) != 1:
                notices['ambiguous_reference' if ids else 'unmatched_reference'] += 1
                obj.setdefault('unresolved_mentions', []).append({'kind': kind, 'label': number,
                                                                   'span': [left, right], 'candidates': ids})
                continue
            edges.append({'paragraph_id': obj['object_id'], 'object_id': ids[0], 'span': [left, right],
                          'type': 'explicit_number', 'label': number})

    incoming, outgoing = defaultdict(set), defaultdict(list)
    for edge in edges:
        incoming[edge['object_id']].add(edge['paragraph_id'])
        outgoing[edge['paragraph_id']].append(edge)

    def add_chunks(obj, text, view, row=None):
        if not text.strip():
            return
        prov = obj['provenance']
        pages = sorted({p['page_no'] for p in prov if str(p['page_no']) in doc['pages']})
        if not pages:
            notices['views_without_page_provenance'] += 1
            return
        prefix = '\n'.join(s for s in (title, ' / '.join(obj['headings'])) if s) + '\n'
        if len(tokenizer.encode(prefix, add_special_tokens=True, truncation=False)) > budget // 4:
            clip = tokenizer(prefix, add_special_tokens=False, truncation=True,
                             max_length=budget//4-3, return_offsets_mapping=True)['offset_mapping']
            prefix = prefix[:clip[-1][1]].rstrip() + '\n' if clip else ''
        for left, right in windows(text, tokenizer, budget, overlap, prefix):
            body = text[left:right]
            if not body.strip():
                continue
            related = {e['object_id'] for e in outgoing[obj['object_id']] if left < e['span'][1] and right > e['span'][0]}
            chunks.append({'chunk_index': len(chunks), 'text': body, 'embedding_text': prefix+body,
                           'tokens': len(tokenizer.encode(prefix+body, add_special_tokens=True, truncation=False)),
                           'headings': obj['headings'], 'kind': {'paragraph': 'text'}.get(obj['kind'], obj['kind']),
                           'elements': [{'id': obj['ref'], 'label': obj['label'], 'provenance': prov}], 'pages': pages,
                           'object_id': obj['object_id'], 'view': view, 'row': row, 'view_span': [left, right],
                           'linked_object_ids': sorted(related), 'linked_paragraph_ids': sorted(incoming[obj['object_id']])})

    for obj in objects:
        if obj['kind'] == 'paragraph':
            add_chunks(obj, obj['text'], 'paragraph')
        elif obj['kind'] == 'figure':
            add_chunks(obj, '\n'.join(filter(None, (obj['caption'], obj['region_text']))), 'figure_text')
        else:
            names = ' | '.join(obj['columns'])
            label = ' | '.join('Table '+n for n in obj['labels'])
            add_chunks(obj, '\n'.join(filter(None, (label, obj['caption'], names or obj['region_text']))), 'table_summary')
            for row in obj['rows']:
                add_chunks(obj, ' | '.join(filter(None, (label, row['text']))), 'table_row', row['row'])
    return objects, edges, chunks, dict(notices)

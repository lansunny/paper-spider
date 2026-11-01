"""Geometry-first formula sidecars; does not change the frozen text index.

Numbered display formulas are anchored by equation labels. Unnumbered/mixed
regions are candidates for review. Native text is never fabricated into LaTeX.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, uuid5

import portalocker
import pymupdf

from rag.chunking import normalized_blocks
from rag.config import ROOT
from rag.formula_schema import FormulaElement, FormulaSourceText, FormulaOCRResult
from rag.storage import atomic_json, atomic_text, read_json

VERSION = 'formula-layout-text-v10'
IDENTITY_VERSION = 'formula-region-id-v1'
NUMBER = re.compile(r'\(((?:[A-Z]\.?\d+|\d+)(?:\.\d+)*[a-h]?)\)\s*$')
RELATIONS = set('=≈≠≤≥∈∉≡∝')
OPERATORS = set('+−*/∑∫∏√∞×÷|^_{}[]<>')
CAPTION = re.compile(r'^(?:Figure|Fig\.|Table|Algorithm)\s+\d', re.I)
MONO = re.compile(r'mono|courier|consol|cmtt|nimbusmonl|typewriter', re.I)
MATH_ROMAN = re.compile(r'^cm(?:r|b|bx|ss|tt)\d|^lmroman', re.I)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def clipped(box, bounds):
    return tuple(float(v) for v in (pymupdf.Rect(box) & pymupdf.Rect(bounds)))


def valid_box(box):
    return len(box) == 4 and box[0] < box[2] and box[1] < box[3]


def math_font(font):
    name = font.split('+')[-1].lower()
    return bool(re.match(r'^cm(?:mib|mi|sy|ex|r|b|bx|bi)\d|^ms(?:am|bm)\d|^mt(?:mi|sy)|^eu(?:fm|fb|fr|sm|sb|ex)|^rsfs', name) or
                any(t in name for t in ['math', 'symbol', 'stix', 'xits', 'asana']))


def non_math_reason(line):
    """Hard negatives are evidence about the source, independent of crops."""
    text = line['body_text']
    chars = [c for c in line['chars'] if not c['c'].isspace()]
    if chars and sum(bool(MONO.search(c['font'])) for c in chars)/len(chars) > .5:
        return 'monospace_code'
    if re.search(r'https?://|www\.|(?:arxiv\.)?abs/\d{4}\.\d+', text, re.I):
        return 'url_or_identifier'
    if re.search(r'\b(?:import|from|def|return)\s+\w|\w+\.(?:evaluate|to_pandas|append|load_dataset)\s*\(|[\w]+\[[\'"]', text):
        return 'programming_syntax'
    if ('_' in text and '/' in text and re.search(r"[\'\"]", text)):
        return 'quoted_path'
    return None


def ygap(a, b):
    return max(0, a[1]-b[3], b[1]-a[3])


def column(box, width, double):
    if not double:
        return 'single'
    mid, tol = width/2, width*.015
    if box[2] <= mid+tol:
        return 'left'
    if box[0] >= mid-tol:
        return 'right'
    return 'full'


def compatible(a, b):
    return a == b or 'full' in (a, b)


def non_formula_layout(page, blocks):
    """Exclude caption-associated ruled tables and vector figures, not whole pages."""
    drawings = page.get_drawings()
    rules, shapes = [], []
    for drawing in drawings:
        rect = tuple(drawing['rect'])
        if rect[2]-rect[0] > 80 and rect[3]-rect[1] < 2:
            rules.append(rect)
        elif rect[2]-rect[0] > 4 and rect[3]-rect[1] > 4:
            shapes.append(rect)
    shapes.extend(tuple(i['bbox']) for i in page.get_image_info())
    exclusions = []
    for caption in blocks:
        match = CAPTION.match(caption['text'])
        if not match:
            continue
        box = caption['bbox']
        if caption['text'].lower().startswith('table'):
            local = [r for r in rules if r[1] < box[1] and r[2] > box[0] and r[0] < box[2]
                     and r[2]-r[0] > (box[2]-box[0])*.65]
            if not local or box[1]-max(r[3] for r in local) > 35:
                continue
            chain = [max(local, key=lambda r: r[3])]
            for r in sorted(local, key=lambda r: r[3], reverse=True)[1:]:
                if min(c[1] for c in chain)-r[3] > min(180, page.rect.height*.25):
                    break
                chain.append(r)
            if len(chain) >= 2:
                exclusions.append(union_box(chain))
        elif caption['text'].lower().startswith(('figure', 'fig.')):
            local = [r for r in shapes if r[3] <= box[1]+3 and r[3] >= box[1]-50
                     and r[2] > box[0] and r[0] < box[2]]
            if not local:
                continue
            envelope = union_box(local)
            while True:
                more = [r for r in shapes if r[3] <= box[1]+3 and r[2] > box[0] and r[0] < box[2]
                        and ygap(envelope, r) <= 20]
                grown = union_box([envelope, *more])
                if grown == envelope:
                    break
                envelope = grown
            if envelope[3]-envelope[1] > 40:
                exclusions.append((min(box[0], envelope[0]), max(0, envelope[1]-12),
                                   max(box[2], envelope[2]), box[1]))
    return exclusions


def page_records(page):
    """Character boxes preserve number positions and mathematical font signals."""
    raw_blocks = [{'block_number': b[5], 'text': b[4], 'bbox': tuple(b[:4])}
                  for b in page.get_text('blocks', sort=False) if b[6] == 0]
    canonical, spans = normalized_blocks(raw_blocks)
    populated = [b for b in raw_blocks if re.sub(r'\s+', ' ', b['text']).strip()]
    blocks = [{'number': b['block_number'], 'text': canonical[a:z], 'start': a, 'end': z, 'bbox': box}
              for b, (a, z, box) in zip(populated, spans, strict=True)]
    caption_blocks = {b['number'] for b in blocks if CAPTION.match(b['text'])}
    exclusions = non_formula_layout(page, blocks)
    lines = []
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    for block in page.get_text('rawdict', flags=flags)['blocks']:
        if block['type'] != 0:
            continue
        for index, line in enumerate(block['lines']):
            if abs(line.get('dir', (1, 0))[1]) > .05:
                continue  # Vertical/rotated labels are not treated as equation bodies.
            chars = [{**c, 'font': s['font'], 'size': s['size']} for s in line['spans'] for c in s['chars']]
            text = ''.join(c['c'] for c in chars)
            if not text.strip():
                continue
            label, label_box, body = None, None, chars
            match = NUMBER.search(text)
            if match and not re.fullmatch(r'(?:19|20)\d{2}', match.group(1)):
                start = match.start()
                spaced = start == 0 or text[start-1].isspace()
                if start and not spaced:
                    spaced = chars[start]['bbox'][0]-chars[start-1]['bbox'][2] > 10
                if spaced:
                    label = match.group(1)
                    label_box = union_box([c['bbox'] for c in chars[start:match.end()] if not c['c'].isspace()])
                    body = chars[:start]
            body_text = ''.join(c['c'] for c in body).strip()
            body_box = union_box([c['bbox'] for c in body if not c['c'].isspace()]) if body_text else None
            in_layout = any(r[0]-2 <= (line['bbox'][0]+line['bbox'][2])/2 <= r[2]+2
                            and r[1]-2 <= (line['bbox'][1]+line['bbox'][3])/2 <= r[3]+2 for r in exclusions)
            lines.append({'id': f"b{block['number']}:l{index}", 'block': block['number'], 'bbox': tuple(line['bbox']),
                          'text': text, 'chars': body, 'body_text': body_text, 'body_bbox': body_box,
                          'number': label, 'number_bbox': label_box, 'caption': block['number'] in caption_blocks or in_layout})
    width = page.rect.width
    prose = [l for l in lines if len(re.findall(r'[A-Za-z]{3,}', l['body_text'])) >= 5]
    double = (sum(l['bbox'][2] <= width/2+width*.015 for l in prose) >= 2 and
              sum(l['bbox'][0] >= width/2-width*.015 for l in prose) >= 2)
    # Map source lines into exactly the normalization used by Chunk.source_spans.
    cursors = {}
    by_number = {b['number']: b for b in blocks}
    for line in lines:
        block = by_number.get(line['block'])
        line_text = re.sub(r'\s+', ' ', line['text']).strip()
        at = block['text'].find(line_text, cursors.get(line['block'], 0)) if block else -1
        if at >= 0:
            line['source_start'], line['source_end'] = block['start']+at, block['start']+at+len(line_text)
            cursors[line['block']] = at+len(line_text)
        line['column'] = column(line['bbox'], width, double)
    for block in blocks:
        block['column'] = column(block['bbox'], width, double)
    return canonical, blocks, lines, double


def signals(line):
    text = line['body_text']
    chars = [c for c in line['chars'] if not c['c'].isspace()]
    share = sum(math_font(c['font']) for c in chars)/max(1, len(chars))
    if non_math_reason(line):
        return False, False, share
    long_words = len(re.findall(r'[A-Za-z]{3,}', text))
    relations = sum(c in RELATIONS for c in text)
    operators = sum(c in OPERATORS for c in text)
    greek = sum('\u0370' <= c <= '\u03ff' for c in text)
    weak_variable = bool(re.fullmatch(r'[A-Za-z](?:[_^]?[0-9]+)?|[0-9]{1,3}|[()\[\]{}|]+', text))
    strong_math_chars = sum(math_font(c['font']) and not MATH_ROMAN.match(c['font'].split('+')[-1]) for c in chars)
    # A table label such as BM25 + DPR is not an arithmetic formula.
    variable_expression = not re.search(r'[A-Za-z]{3,}', text)
    arithmetic = (operators >= 1 and len(text) <= 60 and len(re.findall(r'[A-Za-z]', text)) >= 2
                  and (variable_expression or strong_math_chars >= 2))
    case_condition = bool(re.match(r'^(?:if|when)\b', text, re.I) and (relations or greek) and len(text) <= 80)
    math_function = bool(re.fullmatch(r'[A-Za-z]\w*\([^)]{1,20}\)[,.]?', text) and strong_math_chars >= 2)
    prose = long_words >= 3 and share < .65 and not case_condition
    fragment = bool(text and not line['caption'] and long_words <= 6 and len(text) <= 130
                    and not prose and (long_words <= 4 or share >= .75 or case_condition)
                    and (share >= .35 or relations or greek or operators >= 2 or weak_variable or arithmetic or math_function))
    core = fragment and bool(relations or (share >= .65 and strong_math_chars >= 2 and (len(text) >= 4 or operators))
                             or greek and operators or arithmetic)
    return fragment, core, share


def grow_group(initial, pool, limit_low, limit_high, height=160):
    group = list(initial)
    used = {l['id'] for l in group}
    warnings = []
    while True:
        box = union_box([l['body_bbox'] for l in group])
        added = []
        for line in pool:
            if line['id'] in used:
                continue
            b = line['body_bbox']
            center = (b[1]+b[3])/2
            if not limit_low <= center <= limit_high or ygap(box, b) > 9:
                continue
            proposed = union_box([box, b])
            if proposed[3]-proposed[1] > height or len(group)+len(added) >= 48:
                warnings.append('region_limit_reached')
                continue
            added.append(line)
            used.add(line['id'])
        if not added:
            break
        group.extend(added)
    return group, sorted(set(warnings))


def region_from_group(group, lines, label=None, extra_box=None, page=None):
    group = list(group)
    # English-word numerators/denominators are still part of a numbered fraction.
    # Require a real short horizontal fraction stroke near the detected math body.
    if page is not None and label:
        body = union_box([l['body_bbox'] for l in group])
        for drawing in page.get_drawings():
            stroke = tuple(drawing['rect'])
            if not (12 <= stroke[2]-stroke[0] <= 200 and stroke[3]-stroke[1] < 1.5
                    and ygap(stroke, body) <= 15 and stroke[0] <= body[2]+15 and stroke[2] >= body[0]-15):
                continue
            for line in lines:
                b = line['body_bbox']
                if (b and not line['caption'] and not non_math_reason(line) and not line['number']
                        and b[0] >= stroke[0]-4 and b[2] <= stroke[2]+4 and ygap(b, stroke) <= 12
                        and line['id'] not in {l['id'] for l in group}):
                    group.append(line)
    box = union_box([l['body_bbox'] for l in group] + ([extra_box] if extra_box else []))
    # Recover denominator/superscript glyphs inside nearby prose lines, while
    # avoiding unrelated mathematical symbols farther along the paragraph.
    supplement, supplement_lines = [], {}
    group_ids = {l['id'] for l in group}
    for line in lines:
        if line['id'] in group_ids or line['caption'] or non_math_reason(line) or signals(line)[0]:
            continue
        for char in line['chars']:
            b = char['bbox']
            if (math_font(char['font']) and char['c'].strip() and ygap(box, b) <= 3
                    and b[0] <= box[2]+3 and b[2] >= box[0]-3):
                supplement.append(char)
                supplement_lines[line['id']] = line
    if supplement:
        box = union_box([box, *[c['bbox'] for c in supplement]])
    companions = [l for l in lines if l['id'] not in group_ids and not l['caption'] and not l['number'] and not non_math_reason(l)
                  and len(l['text']) <= 45 and l['bbox'][0] >= box[0] and l['bbox'][2] <= box[2]
                  and l['bbox'][1] >= box[1] and l['bbox'][3] <= box[3]]
    members = [*group, *companions]
    # Only a transcript of source lines; layout remains authoritative in the image.
    text = '\n'.join(l['body_text'] for l in sorted(members, key=lambda l: (l['bbox'][1], l['bbox'][0])) if l['body_text'])
    if supplement:
        text += '\n' + ''.join(c['c'] for c in supplement)
    source_members = {line['id']: line for line in [*members, *supplement_lines.values()]}
    return {'bbox': box, 'number': label, 'lines': list(source_members.values()), 'raw_text': text,
            'warnings': ['math_text_layer_may_be_lossy'] + (['math_glyphs_recovered_from_mixed_lines'] if supplement else [])}


def detect_regions(page):
    canonical, blocks, lines, double = page_records(page)
    pool = [l for l in lines if l['body_bbox'] and signals(l)[0]]
    # A right-aligned parenthesized number alone is not an equation anchor.
    labels = [l for l in lines if l['number'] and not l['caption'] and not non_math_reason(l)
              and (not l['body_text'] or signals(l)[0])]
    typical_height = median([l['bbox'][3]-l['bbox'][1] for l in lines]) if lines else 12
    regions, used, pending = [], set(), []
    for label in sorted(labels, key=lambda l: (l['bbox'][1], l['bbox'][0])):
        b, col = label['number_bbox'], label['column']
        cy = (b[1]+b[3])/2
        # Standalone list markers at the left margin are not equation anchors.
        boundary = page.rect.width/2 if double and col == 'left' else page.rect.width
        origin = page.rect.width/2 if double and col == 'right' else 0
        if (b[0]-origin)/max(1, boundary-origin) < .60:
            continue
        local = [l for l in pool if l['id'] not in used and compatible(col, l['column'])
                 and l['body_bbox'][0] < b[0] and l['body_bbox'][2] <= b[2]]
        initial = [l for l in local if abs((l['body_bbox'][1]+l['body_bbox'][3])/2-cy) <= typical_height*1.4]
        if not initial or not any(signals(l)[1] for l in initial):
            images = [i['bbox'] for i in page.get_image_info() if compatible(col, column(i['bbox'], page.rect.width, double))
                      and i['bbox'][2] <= b[0]+4 and ygap(i['bbox'], b) <= 10
                      and i['bbox'][3]-i['bbox'][1] <= 160]
            if initial or len(images) == 1:
                pending.append({'equation_number': label['number'], 'anchor_bbox': b,
                                'reason': 'equation_body_unresolved',
                                'evidence': 'nearby_math_fragments' if initial else 'nearby_image_not_proof_of_formula'})
            continue
        others = [(l['number_bbox'][1]+l['number_bbox'][3])/2 for l in labels
                  if l['id'] != label['id'] and compatible(col, l['column'])]
        low = max([(y+cy)/2 for y in others if y < cy-2], default=0)
        high = min([(y+cy)/2 for y in others if y > cy+2], default=page.rect.height)
        group, warnings = grow_group(initial, local, low, high)
        region = region_from_group(group, lines, label['number'], b, page)
        region['warnings'].extend(warnings)
        region['column'] = col
        regions.append(region)
        used.update(l['id'] for l in group)
    for seed in sorted(pool, key=lambda l: (l['bbox'][1], l['bbox'][0])):
        if seed['id'] in used or not signals(seed)[1]:
            continue
        local = [l for l in pool if l['id'] not in used and compatible(seed['column'], l['column'])]
        group, warnings = grow_group([seed], local, 0, page.rect.height)
        region = region_from_group(group, lines)
        region['warnings'].extend(warnings)
        region['column'] = seed['column']
        regions.append(region)
        used.update(l['id'] for l in group)
    # Distinguish mixed paragraph expressions from independent display regions.
    for region in regions:
        if region.get('raster'):
            region.update(kind='numbered_raster', confidence='medium', column=column(region['bbox'], page.rect.width, double))
            continue
        ids = {l['id'] for l in region['lines']}
        outside = [l for l in lines if l['block'] in {m['block'] for m in region['lines']}
                   and l['id'] not in ids and not l['number'] and len(re.findall(r'[A-Za-z]{3,}', l['text'])) >= 3]
        region['kind'] = 'numbered_display' if region['number'] else ('inline_or_mixed' if outside else 'unnumbered_display')
        # Detection certainty and presentation scope are independent attributes.
        mathematical = any(signals(l)[1] for l in region['lines'])
        region['confidence'] = 'high' if region['number'] and mathematical else 'medium'
        region['evidence'] = ['math_body'] + (['equation_number'] if region['number'] else [])
        if outside:
            region['warnings'].append('mixed_paragraph_candidate_requires_review')
    return canonical, blocks, sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0])), pending, double


def source_excerpt(block, page, canonical, relation, limit=1100, anchor_at=None):
    text, a, z = block['text'], block['start'], block['end']
    if len(text) > limit:
        if anchor_at is not None:
            a = block['start']+max(0, min(len(text)-limit, anchor_at-limit//3))
            z = a+limit
        elif relation == 'preceding':
            a = z-limit
        else:
            z = a+limit
        text = canonical[a:z]
    return FormulaSourceText(pdf_page=page, block_number=block['number'], bbox=block['bbox'], text=text,
                             source_start=a, source_end=z, canonical_page_sha256=sha_bytes(canonical.encode()), relation=relation)


def nearby_context(region, blocks, page, canonical):
    box, col = region['bbox'], region['column']
    selected = []
    for block in blocks:
        b = block['bbox']
        if not compatible(col, block['column']) or CAPTION.match(block['text']) or ygap(box, b) > 120:
            continue
        relation = 'preceding' if b[3] <= box[1] else ('following' if b[1] >= box[3] else 'overlapping')
        # Keep explanations from intersecting mixed blocks, plus nearest paragraph
        # on each side. Mathematical-only blocks are already in the crop.
        if len(re.findall(r'[A-Za-z]{3,}|[\u4e00-\u9fff]', block['text'])) < 4:
            continue
        selected.append((ygap(box, b), relation, block))
    output = []
    for relation in ['overlapping', 'preceding', 'following']:
        options = sorted((x for x in selected if x[1] == relation), key=lambda x: (x[0], x[2]['number']))
        for _, _, block in options[:2 if relation == 'overlapping' else 1]:
            output.append(source_excerpt(block, page, canonical, relation))
    return output


def formula_source_spans(region, page, canonical):
    """Only glyphs inside the formula box, not an entire supplementary prose line."""
    box = pymupdf.Rect(region['bbox'])
    sources = []
    for line in region['lines']:
        if 'source_start' not in line:
            continue
        positions, cursor = [], 0
        for char in line['chars']:
            if char['c'].strip() and not (pymupdf.Rect(char['bbox']) & box).is_empty:
                positions.append((cursor, cursor+len(char['c']), char['bbox']))
            cursor += len(char['c'])
        if not positions:
            continue
        begin, end = positions[0][0], positions[-1][1]
        a = line['source_start']+len(re.sub(r'\s+', ' ', line['text'][:begin]).lstrip())
        z = line['source_start']+len(re.sub(r'\s+', ' ', line['text'][:end]).lstrip())
        if a >= z or z > line['source_end']:
            continue
        sources.append(FormulaSourceText(pdf_page=page, block_number=line['block'],
            bbox=union_box([p[2] for p in positions]), text=canonical[a:z], source_start=a, source_end=z,
            canonical_page_sha256=sha_bytes(canonical.encode()), relation='formula_body'))
    return sources


def explicit_references(number, all_pages):
    if number is None:
        return []
    pattern = re.compile(r'\b(?:eq(?:uation)?s?|formula)\.?\s*\(?'+re.escape(number)+r'\)?(?![\w.])', re.I)
    result = []
    for page_num, (canonical, blocks) in all_pages.items():
        for block in blocks:
            match = pattern.search(block['text'])
            if match:
                result.append(source_excerpt(block, page_num, canonical, 'explicit_reference', anchor_at=match.start()))
    return result


def extract_document(document, *, root=ROOT, pages=None, dpi=180, progress=None):
    if not 72 <= dpi <= 400:
        raise ValueError('DPI must be between 72 and 400')
    path = Path(document['_path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('PDF outside project')
    digest = sha_bytes(path.read_bytes())
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError('Encrypted PDF')
        chosen = sorted(set(pages if pages is not None else range(1, len(pdf)+1)))
        if not chosen or any(p < 1 or p > len(pdf) for p in chosen):
            raise ValueError('Requested pages are outside this PDF')
        if progress:
            progress('cache_check', 0, len(pdf))
        profile = {'version': VERSION, 'pages': chosen, 'dpi': dpi, 'pymupdf': pymupdf.VersionBind,
                   'merge_gap_points': 9, 'max_region_height_points': 160, 'crop_padding_points': 3,
                   'source_coordinates': 'normalized_blocks-v1',
                   'paper_id': document['paper_id'], 'title': document['title'], 'source_url': document.get('landing_url', ''),
                   'pdf_path': path.relative_to(root).as_posix(),
                   'text_proxy_only': True}
        key = sha_bytes(json.dumps(profile, sort_keys=True).encode())[:12]
        output = root/'runtime/formulas'/digest/key
        output.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(output/'extract.lock'), timeout=0):
            manifest = read_json(output/'manifest.json')
            data_path = output/'elements.jsonl'
            if manifest and data_path.exists() and manifest.get('status') == 'completed':
                try:
                    elements = [FormulaElement.model_validate_json(s) for s in data_path.read_text(encoding='utf-8').splitlines() if s]
                    if (manifest['document_sha256'] == digest and manifest['paper_id'] == document['paper_id']
                            and manifest['profile'] == profile and manifest['elements_sha256'] == sha_bytes(data_path.read_bytes())
                            and manifest['canonical_pages_sha256'] == sha_bytes((output/'canonical_pages.jsonl').read_bytes())
                            and manifest['elements'] == len(elements)
                            and all((root/e.image_path).resolve().is_relative_to(root.resolve())
                                    and (root/e.image_path).is_file()
                                    and sha_bytes((root/e.image_path).read_bytes()) == e.image_sha256 for e in elements)):
                        if progress:
                            progress('cache_hit', len(pdf), len(pdf))
                        return output, elements, {**manifest, 'cache_hit': True}
                except (ValueError, KeyError, OSError):
                    pass
            original_rotations = {i+1: p.rotation for i, p in enumerate(pdf)}
            all_pages, detections, pending, empty = {}, {}, [], []
            for page_num, page in enumerate(pdf, 1):
                page.set_rotation(0)  # In-memory only; no PDF file is saved.
                if page_num in chosen:
                    canonical, blocks, regions, unresolved, _ = detect_regions(page)
                    detections[page_num] = regions
                    pending.extend({'pdf_page': page_num, **item} for item in unresolved)
                    if not canonical.strip():
                        empty.append(page_num)
                else:
                    canonical, blocks, _, _ = page_records(page)
                all_pages[page_num] = (canonical, blocks)
                if progress:
                    progress('detect', page_num, len(pdf))
            labels = Counter(r['number'] for regions in detections.values() for r in regions if r['number'])
            elements = []
            for page_num in chosen:
                page = pdf[page_num-1]
                canonical, blocks = all_pages[page_num]
                for region in detections[page_num]:
                    box = clipped(region['bbox'], page.rect)
                    if not valid_box(box):
                        continue
                    crop_box = box
                    if region['kind'] == 'inline_or_mixed':
                        # Font metrics around inline fractions overlap prose rows.
                        # Keep whole enclosing paragraphs instead of clipping words.
                        enclosing = [b['bbox'] for b in blocks if compatible(region['column'], b['column'])
                                     and set(l['block'] for l in region['lines']).intersection({b['number']})]
                        if enclosing:
                            crop_box = clipped(union_box([box, *enclosing]), page.rect)
                    crop = clipped((crop_box[0]-3, crop_box[1]-3, crop_box[2]+3, crop_box[3]+3), page.rect)
                    identity = [document['paper_id'], digest, IDENTITY_VERSION, page_num,
                                [round(v, 3) for v in box], region['number']]
                    eid = str(uuid5(NAMESPACE_URL, json.dumps(identity)))
                    asset = output/f'p{page_num:03d}_{eid}.png'
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi/72, dpi/72), clip=pymupdf.Rect(crop), alpha=False, annots=False)
                    pix.set_dpi(dpi, dpi)
                    pix.save(asset)
                    context = nearby_context(region, blocks, page_num, canonical)
                    source_spans = formula_source_spans(region, page_num, canonical)
                    refs = explicit_references(region['number'], all_pages)
                    section_candidates = [b for b in blocks if b['bbox'][1] <= box[1] and compatible(region['column'], b['column'])
                                          and len(b['text']) <= 140 and re.match(r'^\d+(?:\.\d+)*\s+[A-Z]', b['text'])]
                    section = max(section_candidates, key=lambda b: b['bbox'][1])['text'] if section_candidates else None
                    warnings = list(region['warnings'])
                    if region['number'] and labels[region['number']] > 1:
                        warnings.append('duplicate_equation_number')
                    if '\ufffd' in region['raw_text']:
                        warnings.append('replacement_glyph_requires_ocr')
                    search = '\n'.join([document['title'], f"Equation {region['number']}" if region['number'] else 'Unnumbered formula',
                                        section or '', region['raw_text'], *[c.text for c in context]])
                    elements.append(FormulaElement(element_id=eid, paper_id=document['paper_id'], document_sha256=digest,
                        title=document['title'], source_url=document.get('landing_url', ''), pdf_page=page_num,
                        page_width=page.rect.width, page_height=page.rect.height, page_rotation=original_rotations[page_num],
                        bbox=box, crop_bbox=crop,
                        crop_scope='mixed_paragraph_context' if region['kind'] == 'inline_or_mixed' else 'formula_region',
                        equation_number=region['number'], layout_kind=region['kind'],
                        detection_method='equation_number_and_raster' if region.get('raster') else 'character_fonts_and_geometry',
                        confidence_band=region['confidence'], source_line_ids=[l['id'] for l in region['lines']],
                        detection_evidence=region.get('evidence', []), source_spans=source_spans,
                        canonical_page_sha256=sha_bytes(canonical.encode()),
                        raw_text=region['raw_text'], context=context, references=refs,
                        reference_resolution=('not_numbered' if region['number'] is None else
                                              'ambiguous' if labels[region['number']] > 1 else 'unique_in_extracted_pages'),
                        section_title=section, search_text=search, image_path=asset.relative_to(root).as_posix(),
                        image_sha256=sha_bytes(asset.read_bytes()), render_dpi=dpi, warnings=sorted(set(warnings))))
                if progress:
                    progress('render', page_num, len(pdf))
            text = ''.join(e.model_dump_json()+'\n' for e in elements)
            atomic_text(data_path, text)
            page_rows = [{'pdf_page': p, 'text': text, 'canonical_page_sha256': sha_bytes(text.encode()),
                          'blocks': blocks} for p, (text, blocks) in all_pages.items()]
            atomic_text(output/'canonical_pages.jsonl', ''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in page_rows))
            jobs = [{'element_id': e.element_id, 'paper_id': e.paper_id, 'document_sha256': digest,
                     'image_path': e.image_path, 'image_sha256': e.image_sha256,
                     'crop_scope': e.crop_scope, 'target_bbox_in_page_points': e.bbox,
                     'crop_bbox_in_page_points': e.crop_bbox, 'render_dpi': dpi,
                     'expected_output': 'LaTeX transcription only; do not infer meaning',
                     'requires_review': True} for e in elements]
            atomic_text(output/'ocr_jobs.jsonl', ''.join(json.dumps(j, ensure_ascii=False)+'\n' for j in jobs))
            layout_jobs = [{'paper_id': document['paper_id'], 'document_sha256': digest,
                            'pdf_path': path.relative_to(root).as_posix(), **item}
                           for item in [*pending, *[{'pdf_page': p, 'reason': 'no_text_layer_layout_detection_required'} for p in empty]]]
            atomic_text(output/'layout_jobs.jsonl', ''.join(json.dumps(j, ensure_ascii=False)+'\n' for j in layout_jobs))
            manifest = {'status': 'completed', 'paper_id': document['paper_id'], 'document_sha256': digest,
                        'pdf_path': path.relative_to(root).as_posix(), 'profile': profile, 'selected_pages': chosen, 'page_count': len(pdf),
                        'identity_version': IDENTITY_VERSION,
                        'elements': len(elements), 'numbered': sum(e.equation_number is not None for e in elements),
                        'elements_sha256': sha_bytes(data_path.read_bytes()), 'pending_layout': pending,
                        'canonical_pages_sha256': sha_bytes((output/'canonical_pages.jsonl').read_bytes()),
                        'pages_needing_ocr_layout': empty, 'latex_recognition_performed': False,
                        'limitations': 'Heuristic candidates, not verified detection recall. Unnumbered inline expressions, diagrams and tables can be ambiguous. Scanned/vector-outline-only formulas need an external layout/OCR provider. Raw glyph order is not mathematical structure.'}
            atomic_json(output/'manifest.json', manifest)
            return output, elements, {**manifest, 'cache_hit': False}


def import_ocr_results(directory, results, root=ROOT):
    """Keep OCR enrichment separate; reject stale or unrelated crop results."""
    manifest = read_json(directory/'manifest.json')
    data = directory/'elements.jsonl'
    if not manifest or manifest['elements_sha256'] != sha_bytes(data.read_bytes()):
        raise ValueError('Formula extraction is incomplete or changed')
    elements = {e.element_id: e for e in (FormulaElement.model_validate_json(s) for s in data.read_text(encoding='utf-8').splitlines() if s)}
    incoming, seen = [], set()
    for value in results:
        result = FormulaOCRResult.model_validate(value)
        if result.element_id in seen:
            raise ValueError('Duplicate OCR result IDs')
        seen.add(result.element_id)
        e = elements.get(result.element_id)
        if not e or e.document_sha256 != result.document_sha256 or e.image_sha256 != result.image_sha256:
            raise ValueError('OCR result does not match the source document and exact crop')
        asset = (root/e.image_path).resolve()
        if not asset.is_relative_to(root.resolve()) or sha_bytes(asset.read_bytes()) != e.image_sha256:
            raise ValueError('Formula crop was changed')
        incoming.append(result.model_dump())
    existing_path = directory/'recognitions.jsonl'
    with portalocker.Lock(str(directory/'recognition.lock'), timeout=10):
        existing = {r['element_id']: r for r in (json.loads(s) for s in existing_path.read_text(encoding='utf-8').splitlines() if s)} if existing_path.exists() else {}
        existing.update({r['element_id']: r for r in incoming})
        atomic_text(existing_path, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in existing.values()))
    return len(incoming)

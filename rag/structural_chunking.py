"""Conservative layout/block chunking with immutable original-page provenance.

No text is deleted. Headings, tables, formulas and references remain searchable.
Layout is a geometry heuristic: full-width anchors split horizontal bands;
within a band, read left column then right. One-column pages use vertical order.
"""
import re

from rag.chunking import normalized_blocks
from rag.schema import Chunk, SourceSpan


def ordered_blocks(page):
    canonical, spans = normalized_blocks(page['blocks'])
    nonempty = [b for b in page['blocks'] if re.sub(r'\s+', ' ', b['text']).strip()]
    records = [{'start': start, 'end': end, 'bbox': box, 'number': b['block_number'],
                'text': canonical[start:end]} for b, (start, end, box) in zip(nonempty, spans, strict=True)]
    width = page['page_width']
    mid, tolerance = width / 2, width * .015

    def column(r):
        x0, _, x1, _ = r['bbox']
        if x0 < mid - tolerance and x1 > mid + tolerance:
            return 'full'
        if x1 <= mid + tolerance:
            return 'left'
        if x0 >= mid - tolerance:
            return 'right'
        return 'full'

    # Avoid imposing two columns on title pages/single-column appendices.
    left = sum(column(r) == 'left' and len(r['text']) >= 80 for r in records)
    right = sum(column(r) == 'right' and len(r['text']) >= 80 for r in records)
    double = left >= 2 and right >= 2
    if not double:
        return canonical, [{**r, 'column': 'single', 'region': 0}
                           for r in sorted(records, key=lambda r: (r['bbox'][1], r['bbox'][0], r['number']))]
    full = sorted([r for r in records if column(r) == 'full'], key=lambda r: (r['bbox'][1], r['number']))
    rest = [r for r in records if column(r) != 'full']
    result, region = [], 0

    def emit(items, band):
        return [{**r, 'column': column(r), 'region': band} for r in sorted(
            items, key=lambda r: (0 if column(r) == 'left' else 1, r['bbox'][1], r['bbox'][0], r['number']))]

    for anchor in full:
        before = [r for r in rest if (r['bbox'][1] + r['bbox'][3]) / 2 < anchor['bbox'][1]]
        result.extend(emit(before, region))
        done = {r['number'] for r in before}
        rest = [r for r in rest if r['number'] not in done]
        region += 1
        result.append({**anchor, 'column': 'full', 'region': region})
        region += 1
    result.extend(emit(rest, region))
    return canonical, result


def token_count(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])


def block_pieces(record, tokenizer, budget):
    """Only split oversized blocks. Prefer sentence ends, token fallback if needed.

    Slice original characters, rather than decoding tokens, so source alignment
    survives ligatures and multi-token Unicode characters. No token overlap.
    """
    text = record['text']
    if token_count(text, tokenizer) <= budget:
        yield {**record, 'split_method': 'whole-block'}
        return
    start = 0
    sentence_ends = [m.end() for m in re.finditer(r'[.!?。！？](?=\s|$)', text)]
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            return
        remaining = text[start:]
        encoded = tokenizer(remaining, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = encoded['offset_mapping']
        if len(offsets) <= budget:
            end, method = len(text), 'sentence-split'
        else:
            limit = start + offsets[budget - 1][1]
            boundaries = [p for p in sentence_ends if start < p <= limit]
            end, method = (boundaries[-1], 'sentence-split') if boundaries else (limit, 'token-fallback')
            # Re-tokenization at a new boundary can change token counts.
            while end > start and token_count(text[start:end].strip(), tokenizer) > budget:
                end -= 1
                method = 'token-fallback'
        if end <= start:
            raise ValueError('Unable to split oversized block within token budget')
        actual_end = end
        while actual_end > start and text[actual_end - 1].isspace():
            actual_end -= 1
        yield {**record, 'start': record['start'] + start, 'end': record['start'] + actual_end,
               'text': text[start:actual_end], 'split_method': method}
        start = end


def is_heading(text):
    return len(text) <= 140 and bool(re.match(
        r'^(?:(?:\d+(?:\.\d+)*|[A-Z])\s+[A-Z]|(?:Abstract|Introduction|Conclusion[s]?|References|Acknowledg\w+|Limitations|Appendix)\s*$)', text))


def structural_chunks(page, tokenizer, settings, **metadata):
    _, records = ordered_blocks(page)
    parts, ordinal, section = [], 0, ''

    def make_chunk(pieces, number):
        texts, source_spans, cursor = [], [], 0
        for part in pieces:
            if texts:
                cursor += 2
            source_spans.append(SourceSpan(block_number=part['number'], source_start=part['start'],
                                           source_end=part['end'], text_start=cursor, text_end=cursor + len(part['text'])))
            texts.append(part['text'])
            cursor += len(part['text'])
        methods = {p['split_method'] for p in pieces}
        method = 'token-fallback' if 'token-fallback' in methods else ('sentence-split' if 'sentence-split' in methods else 'whole-block')
        return Chunk(**metadata, chunker_version=settings.chunker_version, pdf_page=page['pdf_page'],
                     chunk_index=number, text='\n\n'.join(texts),
                     bboxes=tuple(p['bbox'] for p in pieces), source_spans=tuple(source_spans),
                     layout_column=pieces[0]['column'], layout_region=pieces[0]['region'],
                     section_title=pieces[0]['section'], split_method=method)

    for record in records:
        heading = is_heading(record['text'])
        if heading:
            section = record['text']
        for part in block_pieces(record, tokenizer, settings.chunk_size):
            part['section'] = section
            merge = False
            if parts and not heading:
                prev = parts[-1]
                a, b = prev['bbox'], part['bbox']
                overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                width = max(.001, min(a[2] - a[0], b[2] - b[0]))
                small = min(token_count(prev['text'], tokenizer), token_count(part['text'], tokenizer)) <= settings.small_block_tokens
                merge = (prev['column'] == part['column'] and prev['region'] == part['region']
                         and prev['number'] != part['number'] and small
                         and -2 <= b[1] - a[3] <= settings.block_merge_gap and overlap / width > .5
                         and token_count('\n\n'.join(p['text'] for p in [*parts, part]), tokenizer) <= settings.chunk_size)
            if parts and not merge:
                yield make_chunk(parts, ordinal)
                ordinal += 1
                parts = []
            parts.append(part)
    if parts:
        yield make_chunk(parts, ordinal)

"""Merge overlapping source spans without inventing text or using gold labels."""
import hashlib
import re
from collections import defaultdict


def merge_source_windows(hits, source_pages):
    """Union overlapping/adjacent spans within one immutable document page.

    Windows keep the priority of their earliest input hit. Disjoint spans remain
    separate; no omitted text is silently filled in. These are context windows,
    not newly indexed chunks or newly scored retrieval hits.
    """
    grouped = defaultdict(list)
    for priority, hit in enumerate(hits):
        key = (hit['paper_id'], hit['document_sha256'], hit['pdf_page'])
        page = source_pages[key]
        text = hit['text'].strip()
        start = page.find(text)
        if not text or start < 0 or page.find(text, start + 1) >= 0:
            raise ValueError(f'Cannot uniquely align chunk to source page: {key}')
        grouped[key].append({'start': start, 'end': start + len(text),
                             'priority': priority,
                             'ids': set(hit['source_chunk_ids']) if 'source_chunk_ids' in hit else {hit['chunk_id']},
                             'hit': hit})
    windows = []
    for key, spans in grouped.items():
        merged = []
        for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
            if merged and span['start'] <= merged[-1]['end']:
                previous = merged[-1]
                previous['end'] = max(previous['end'], span['end'])
                previous['ids'].update(span['ids'])
                if span['priority'] < previous['priority']:
                    previous['priority'], previous['hit'] = span['priority'], span['hit']
            else:
                merged.append({**span, 'ids': set(span['ids'])})
        for span in merged:
            identity = repr((*key, span['start'], span['end']))
            windows.append({'window_id': hashlib.sha256(identity.encode()).hexdigest()[:24],
                            'paper_id': key[0], 'document_sha256': key[1], 'pdf_page': key[2],
                            'title': span['hit'].get('title', ''),
                            'source_url': span['hit'].get('source_url', ''),
                            'char_start': span['start'], 'char_end': span['end'],
                            'text': source_pages[key][span['start']:span['end']],
                            'priority': span['priority'], 'source_chunk_ids': sorted(span['ids'])})
    return sorted(windows, key=lambda w: (w['priority'], w['paper_id'], w['pdf_page'], w['char_start']))


def bounded_neighbor_windows(seeds, chunks, source_pages, max_seed_chars=2400):
    """Keep each complete seed, add nearby text up to a per-seed body budget.

    Expansion stays inside overlapping immediate neighbors on the same page.
    Split spare characters between left/right; shrink extensions to punctuation
    or paragraph boundaries. This is a heuristic, not linguistic sentence parsing.
    No gold question, answer or evidence span participates in selection.
    Returned windows are not merged yet: merging may exceed the per-seed cap.
    """
    if max_seed_chars <= 0:
        raise ValueError('max_seed_chars must be positive')
    lookup = {(c.paper_id, c.document_sha256, c.pdf_page, c.chunk_index): c for c in chunks}
    result = []
    for rank, seed in enumerate(seeds):
        key = (seed['paper_id'], seed['document_sha256'], seed['pdf_page'])
        page = source_pages[key]
        original = merge_source_windows([seed], source_pages)[0]
        start, end = original['char_start'], original['char_end']
        if end - start > max_seed_chars:
            raise ValueError('Seed exceeds per-seed budget; increase it instead of silently truncating the seed.')
        neighbors = [seed]
        for delta in [-1, 1]:
            neighbor = lookup.get((*key, seed['chunk_index'] + delta))
            if neighbor is not None:
                neighbors.append(neighbor.payload())
        available = next(w for w in merge_source_windows(neighbors, source_pages)
                         if w['char_start'] <= start and w['char_end'] >= end)
        extra = max_seed_chars - (end - start)
        left_extra = min(extra // 2, start - available['char_start'])
        right_extra = min(extra - left_extra, available['char_end'] - end)
        left_extra = min(extra - right_extra, start - available['char_start'])
        left, right = start - left_extra, end + right_extra
        boundaries = [0] + [m.end() for m in re.finditer(r'[.!?。！？](?=\s|$)|\n\n', page)] + [len(page)]
        left = min((p for p in boundaries if left <= p <= start), default=start)
        right = max((p for p in boundaries if end <= p <= right), default=end)
        # Trim whitespace while preserving exact canonical source coordinates.
        while left < start and page[left].isspace():
            left += 1
        while right > end and page[right - 1].isspace():
            right -= 1
        text = page[left:right]
        source_ids = []
        for hit in neighbors:
            aligned = merge_source_windows([hit], source_pages)[0]
            if aligned['char_start'] < right and aligned['char_end'] > left:
                source_ids.append(hit['chunk_id'])
        result.append({**original, 'window_id': hashlib.sha256(repr((*key, left, right)).encode()).hexdigest()[:24],
                       'char_start': left, 'char_end': right, 'text': text,
                       'priority': rank, 'source_chunk_ids': sorted(set(source_ids))})
    return result

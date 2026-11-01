"""Chunk-independent evidence coverage for the local, agent-annotated benchmark.

Evidence-unit coverage measures labeled source spans, not answer correctness.
Use the exact chunks supplied to the generator. Gold paper IDs are evaluation
metadata and must never be passed as retrieval filters.
"""
import hashlib
import json
from collections import defaultdict

from rag.chunking import normalized_blocks
from rag.config import ROOT
from rag.ingestion import paper_catalog, pdf_pages


def load_questions(split='dev', root=ROOT):
    if split not in {'dev', 'heldout'}:
        raise ValueError('split must be dev or heldout')
    path = root / 'data/evaluation/paper_benchmark' / f'{split}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def load_source_pages(paper_ids, root=ROOT):
    """Read canonical pages from PDFs/cache and verify their actual file hashes."""
    wanted = set(paper_ids)
    catalog = {p['paper_id']: p for p in paper_catalog(root)}
    if wanted - catalog.keys():
        raise ValueError(f'Missing source papers: {sorted(wanted - catalog.keys())}')
    pages = {}
    for pid in sorted(wanted):
        document = catalog[pid]
        path = (root / document['pdf_path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('PDF outside project')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source = {**document, '_path': path, 'sha256': digest}
        for page in pdf_pages(source, root):
            text, _ = normalized_blocks(page['blocks'])
            pages[(pid, digest, page['pdf_page'])] = text
    return pages


def interval_coverage(start, end, intervals):
    """Union intersecting ranges; overlapping chunks must not double-count."""
    if end <= start:
        raise ValueError('Evidence span must have positive length')
    ranges = sorted((max(start, a), min(end, b)) for a, b in intervals if b > start and a < end)
    covered, cursor = 0, start
    for left, right in ranges:
        if right > cursor:
            covered += right - max(left, cursor)
            cursor = right
    return covered / (end - start)


def chunk_intervals(chunks, source_pages):
    """Require unchanged contiguous source text and unambiguous source offsets.

    Unknown/irrelevant sources cannot cover gold. Changed hashes for known papers
    fail explicitly. Prefixing the title for reranking does not alter chunk.text;
    pass the original body, not the model's synthetic title+body input.
    """
    result = defaultdict(list)
    known = {key[0] for key in source_pages}
    for chunk in chunks:
        payload = chunk.model_dump() if hasattr(chunk, 'model_dump') else chunk
        key = (payload['paper_id'], payload['document_sha256'], payload['pdf_page'])
        if key[0] not in known:
            continue
        if key not in source_pages:
            raise ValueError(f'Unknown hash or page for benchmark source: {key}')
        if payload.get('source_spans'):
            text, page, cursor = payload['text'], source_pages[key], 0
            for span in payload['source_spans']:
                a, b = span['source_start'], span['source_end']
                left, right = span['text_start'], span['text_end']
                if not (0 <= a < b <= len(page) and cursor <= left < right <= len(text)):
                    raise ValueError('Invalid structural source-span coordinates')
                if text[cursor:left].strip() or text[left:right] != page[a:b]:
                    raise ValueError('Structural text does not match immutable source span')
                result[key].append((a, b))
                cursor = right
            if text[cursor:].strip():
                raise ValueError('Unmapped structural text')
            continue
        text = payload['text'].strip()
        if not text:
            raise ValueError('Empty retrieved source text')
        page = source_pages[key]
        start = page.find(text)
        if start < 0 or page.find(text, start + 1) >= 0:
            raise ValueError(f'Chunk is not a unique exact source substring: {key}')
        result[key].append((start, start + len(text)))
    # Original block separators are whitespace, not evidence that needs retrieval.
    # Bridge them only when the source text on BOTH sides has been retrieved.
    for key, intervals in result.items():
        merged = []
        for a, b in sorted(intervals):
            if merged and (a <= merged[-1][1] or not source_pages[key][merged[-1][1]:a].strip()):
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        result[key] = merged
    return result


def score_evidence(question, chunks, source_pages, *, k=5, min_coverage=1.0):
    """AND spans in an alternative, OR alternatives, macro-average required units.

    All required units must be covered for all_required_evidence=True. Unsupported
    questions intentionally return null retrieval scores. For open questions,
    these are anchor-coverage scores, not exhaustive relevance/precision labels.
    """
    if k <= 0 or not 0 < min_coverage <= 1:
        raise ValueError('k must be positive and min_coverage in (0, 1]')
    if not question['answerable']:
        return {'question_id': question['question_id'], 'retrieval_eligible': False,
                'evidence_unit_recall': None, 'all_required_evidence': None, 'units': []}
    units = question['evidence_units']
    if not units:
        raise ValueError('Answerable question without evidence')
    intervals = chunk_intervals(list(chunks)[:k], source_pages)
    results = []
    for unit in units:
        alternatives = []
        for alternative in unit['alternatives']:
            coverages = []
            for span in alternative:
                key = (span['paper_id'], span['document_sha256'], span['pdf_page'])
                if key not in source_pages:
                    raise ValueError(f'Missing annotated source: {key}')
                if source_pages[key][span['char_start']:span['char_end']] != span['quote']:
                    raise ValueError('Gold span does not match source')
                coverages.append(interval_coverage(span['char_start'], span['char_end'], intervals[key]))
            alternatives.append(coverages)
        hit = any(all(c + 1e-12 >= min_coverage for c in group) for group in alternatives)
        results.append({'unit_id': unit['unit_id'], 'covered': hit, 'alternative_span_coverages': alternatives})
    count = sum(u['covered'] for u in results)
    return {'question_id': question['question_id'], 'retrieval_eligible': True, 'k': k,
            'min_span_coverage': min_coverage, 'evidence_unit_recall': count / len(results),
            'all_required_evidence': count == len(results), 'units': results,
            'sources_exhaustive': question['gold_sources_exhaustive']}

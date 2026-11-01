"""Offline boundary ablation: local tokenizer only, no vectors or generation.

Compare existing layout blocks with relaxed paragraph merging. A deterministic
paper sample measures size distribution; dev evidence pages diagnose boundaries.
This is not a retrieval evaluation and never reads heldout annotations.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import chunk_intervals, interval_coverage, load_questions
from rag.chunking import normalized_blocks
from rag.config import load_settings, RetrievalSettings
from rag.ingestion import current_manifest, pdf_pages, source_documents
from rag.storage import atomic_json
from rag.structural_chunking import structural_chunks, token_count


def length_summary(values):
    values = sorted(values)
    return {'chunks': len(values), 'tokens_total': sum(values),
            'p50': values[len(values)//2], 'p90': values[int(len(values)*.9)],
            'max': max(values), 'under_64_fraction': sum(n < 64 for n in values)/len(values)}


def check_page(chunks, page, key, cfg, tokenizer):
    text = normalized_blocks(page['blocks'])[0]
    sources = {key: text}
    intervals = chunk_intervals(chunks, sources).get(key, [])
    cursor = 0
    for a, b in intervals:
        if text[cursor:a].strip():
            raise ValueError(f'Lost source text: {key}')
        cursor = max(cursor, b)
    if text[cursor:].strip():
        raise ValueError(f'Lost trailing source text: {key}')
    lengths = [token_count(c.text, tokenizer) for c in chunks]
    if any(n > cfg.chunk_size for n in lengths):
        raise ValueError(f'Token overflow: {key}')
    return sources, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=20, help='0=all papers for size distribution')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/retrieval_structure.json')
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error('sample-size must be nonnegative')
    base = load_settings(args.config)
    if base.chunk_strategy != 'layout-blocks':
        raise ValueError('Expected an existing layout-blocks configuration')
    configs = {'current': base,
               'merge256': RetrievalSettings(**{**base.model_dump(), 'small_block_tokens': 256}),
               'merge256_gap20': RetrievalSettings(**{**base.model_dump(), 'small_block_tokens': 256,
                                                     'block_merge_gap': 20})}
    documents = source_documents(base)
    manifest = current_manifest(base)
    hashes = {d['paper_id']: d['sha256'] for d in documents}
    if manifest is None or manifest['documents'] != hashes or manifest['profile'] != base.index_profile():
        raise ValueError('Source/config no longer matches the existing structural snapshot')
    # Selection depends only on IDs, not questions, evidence or previous failures.
    sample = sorted(hashes, key=lambda pid: hashlib.sha256(pid.encode()).hexdigest())
    if args.sample_size:
        sample = sample[:args.sample_size]
    sample = set(sample)
    gold = defaultdict(dict)
    for q in load_questions('dev'):
        if not q['answerable']:
            continue
        for unit in q['evidence_units']:
            for alternative in unit['alternatives']:
                for span in alternative:
                    key = (span['paper_id'], span['document_sha256'], span['pdf_page'])
                    if hashes.get(key[0]) != key[1]:
                        raise ValueError('Development annotation uses a different PDF')
                    identity = (span['char_start'], span['char_end'])
                    entry = gold[key].setdefault(identity, {'span': span, 'questions': []})
                    if q['question_id'] not in entry['questions']:
                        entry['questions'].append(q['question_id'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/base.embedding_path), local_files_only=True, use_fast=True)
    tokenizer.model_max_length = 10**9
    lengths = defaultdict(list)
    span_results, processed = [], 0
    needed = sample | {key[0] for key in gold}
    for doc in documents:
        pid = doc['paper_id']
        if pid not in needed:
            continue
        for page in pdf_pages(doc):
            key = (pid, doc['sha256'], page['pdf_page'])
            if pid not in sample and key not in gold:
                continue
            per_span = {identity: {'paper_id': pid, 'page': page['pdf_page'],
                                  'start': identity[0], 'end': identity[1],
                                  'questions': item['questions'], 'variants': {}}
                        for identity, item in gold.get(key, {}).items()}
            for label, cfg in configs.items():
                chunks = list(structural_chunks(page, tokenizer, cfg, paper_id=pid,
                                                document_sha256=doc['sha256'], title=doc['title']))
                sources, counts = check_page(chunks, page, key, cfg, tokenizer)
                if pid in sample:
                    lengths[label].extend(counts)
                if per_span:
                    single_ranges = [chunk_intervals([c], sources).get(key, []) for c in chunks]
                    all_ranges = chunk_intervals(chunks, sources).get(key, [])
                    for identity, row in per_span.items():
                        a, b = identity
                        if sources[key][a:b] != gold[key][identity]['span']['quote']:
                            raise ValueError('Annotation text mismatch')
                        if interval_coverage(a, b, all_ranges) < 1-1e-12:
                            raise ValueError('Full page cannot reconstruct evidence')
                        fractions = [interval_coverage(a, b, spans) for spans in single_ranges]
                        row['variants'][label] = {
                            'best_single_fraction': max(fractions, default=0),
                            'overlapping_chunks': sum(f > 0 for f in fractions)}
            span_results.extend(per_span.values())
        processed += 1
        if processed % 10 == 0:
            print(f'Checked {processed}/{len(needed)} documents', flush=True)
    target = ROOT/'runtime/experiments'/('boundaries-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    report = {'status': 'completed', 'source_snapshot': manifest['snapshot_id'],
              'sample_ids': sorted(sample), 'documents_sha256': {pid: hashes[pid] for pid in sorted(needed)},
              'sample_note': 'Size statistics use the ID-hash sample only; dev evidence pages are scored separately.',
              'variants': {label: {'settings': cfg.model_dump(), 'sample_lengths': length_summary(lengths[label]),
                                  'single_chunk_spans': sum(r['variants'][label]['best_single_fraction'] >= 1-1e-12
                                                           for r in span_results)} for label, cfg in configs.items()},
              'unique_dev_spans': len(span_results), 'span_results': span_results,
              'validation': 'Checked exact source alignment, all nonwhitespace text retained, token bounds and full evidence reconstructability on processed pages.',
              'limitation': 'Boundary diagnostics only, not recall or answer accuracy; no index built or changed. Variant settings are not deployment recommendations.'}
    atomic_json(target/'results.json', report)
    for label, row in report['variants'].items():
        print(label, json.dumps(row['sample_lengths']), f"single-span={row['single_chunk_spans']}/{len(span_results)}")
    print('Result:', target/'results.json')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

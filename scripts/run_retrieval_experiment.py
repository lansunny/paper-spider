"""Run a controlled retrieval experiment on the annotated development questions.

PyCharm: use .venv-rag and run this file; the default is A on one question.
CLI: --variants A --limit 0 runs A on all answerable development questions.
No generation API, index writes, or heldout questions are used.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.embeddings import LocalModels
from rag.ingestion import current_chunks
from rag.retrieval import LexicalIndex, reciprocal_rank_fusion, rerank_candidates
from rag.storage import atomic_json
from rag.vector_store import QdrantStore


VARIANTS = {
    'A': {'bm25_language': 'zh', 'candidate_k': 30, 'rerank_title': False},
    'B': {'bm25_language': 'en', 'candidate_k': 30, 'rerank_title': False},
    'C': {'bm25_language': 'en', 'candidate_k': 100, 'rerank_title': False},
    'D': {'bm25_language': 'en', 'candidate_k': 100, 'rerank_title': True},
}
STAGES = ['bm25', 'dense', 'union', 'rrf20', 'before_top5', 'after_top5']


def evaluate_variant(question, branches, models, pages, variant):
    spec = VARIANTS[variant]
    bm25 = branches[f"bm25_{spec['bm25_language']}"][:spec['candidate_k']]
    dense = branches['dense'][:spec['candidate_k']]
    merged = reciprocal_rank_fusion({'bm25': bm25, 'dense': dense}, k=60)
    candidates = merged[:20]
    start = perf_counter()
    ranked = rerank_candidates(models, question['question'], candidates, title=spec['rerank_title'])
    elapsed = (perf_counter() - start) * 1000
    lists = {'bm25': bm25, 'dense': dense, 'union': merged, 'rrf20': candidates,
             'before_top5': candidates[:5], 'after_top5': ranked[:5]}
    metrics = {stage: score_evidence(question, hits, pages, k=max(1, len(hits)))
               for stage, hits in lists.items()}
    # Store bodies and scores to inspect false positives and missing evidence later.
    return {'question_id': question['question_id'], 'question': question['question'],
            'scope': question['source_scope'], 'category': question['category'],
            'variant': variant, 'variant_settings': spec,
            'bm25_query': question['question_en'] if spec['bm25_language'] == 'en' else question['question'],
            'dense_query': question['question'], 'rerank_query': question['question'],
            'counts': {stage: len(hits) for stage, hits in lists.items()},
            'metrics': metrics, 'rankings': lists,
            'reranked_candidates': ranked,
            'rerank_ms_including_first_load': round(elapsed, 2)}


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[(row['variant'], 'all_answerable')].append(row)
        groups[(row['variant'], row['scope'])].append(row)
    output = []
    for (variant, scope), rows in sorted(groups.items()):
        output.append({'variant': variant, 'scope': scope, 'questions': len(rows),
                       'stages': {stage: {
                           'mean_evidence_unit_recall': sum(r['metrics'][stage]['evidence_unit_recall'] for r in rows) / len(rows),
                           'all_required_evidence_rate': sum(r['metrics'][stage]['all_required_evidence'] for r in rows) / len(rows),
                       } for stage in STAGES}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Independent retrieval configuration JSON')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['A'])
    parser.add_argument('--limit', type=int, default=1, help='0 = all answerable dev questions; default 1')
    parser.add_argument('--question-id', help='Run one specific answerable development question')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    variants = list(dict.fromkeys(args.variants))
    all_questions = load_questions('dev')
    questions = [q for q in all_questions if q['answerable']]
    if args.question_id:
        questions = [q for q in questions if q['question_id'] == args.question_id]
        if not questions:
            parser.error('question-id must be an answerable question in dev')
    elif args.limit:
        questions = questions[:args.limit]
    settings = load_settings(args.config)
    if settings.corpus != 'papers':
        raise ValueError('This experiment requires the papers corpus, not the QASPER index.')
    manifest, chunks = current_chunks(settings)
    paper_ids = {pid for q in questions for pid in q['gold_paper_ids']}
    # These IDs load gold source texts only; no retrieval call receives a paper filter.
    pages = load_source_pages(paper_ids)
    for pid, digest, _ in pages:
        if manifest['documents'].get(pid) != digest:
            raise ValueError(f'Indexed source differs from annotated PDF: {pid}')
    directory = ROOT / 'runtime/experiments' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:8])
    directory.mkdir(parents=True)
    provenance = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                  'snapshot_id': manifest['snapshot_id'], 'index_profile': manifest['profile'],
                  'settings': settings.model_dump(), 'split': 'dev', 'variants': variants,
                  'question_ids': [q['question_id'] for q in questions],
                  'skipped_unsupported_questions': [q['question_id'] for q in all_questions if not q['answerable']],
                  'benchmark_sha256': hashlib.sha256((ROOT / 'data/evaluation/paper_benchmark/dev.jsonl').read_bytes()).hexdigest(),
                  'fixed_controls': {'dense_language': 'zh', 'dense_search_limit': 100,
                      'dense_exact': False, 'rrf_k': 60, 'rerank_k': 20, 'final_k': 5,
                      'minimum_span_coverage': 1.0, 'gold_paper_filter': False},
                  'interpretation': 'Coverage of annotated evidence anchors, not answer accuracy. Unnamed-source questions are reported separately because other correct evidence may exist.',
                  'timing_note': 'Shared retrieval branches and model loading make these timings diagnostic, not a controlled latency benchmark.',
                  'english_query_note': 'Fixed source-authored translations are a controlled diagnostic; their gains do not measure an automatic query-rewriting system.'}
    atomic_json(directory / 'run.json', provenance)
    print(f"索引：{manifest['snapshot_id']}；论文：{len(manifest['documents'])}；分块：{len(chunks)}", flush=True)
    print(f"开发集本次运行 {len(questions)} 题，方案 {', '.join(variants)}；结果目录：{directory}", flush=True)
    store = QdrantStore(settings)
    try:
        store.validate_collection()
        if store.count(manifest['snapshot_id']) != len(chunks):
            raise ValueError('Qdrant snapshot count does not match the local index.')
        lexical, models, results = LexicalIndex(chunks), LocalModels(settings), []
        for number, question in enumerate(questions, 1):
            print(f"[{number}/{len(questions)}] {question['question_id']} {question['question']}", flush=True)
            branches = {}
            for language in sorted({VARIANTS[v]['bm25_language'] for v in variants}):
                query = question['question_en'] if language == 'en' else question['question']
                branches[f'bm25_{language}'] = lexical.search(query, top_k=100)
            # All variants use prefixes of the same dense ranking for this question.
            vector = models.query_vector(question['question'])
            branches['dense'] = store.search(vector, manifest['snapshot_id'], top_k=100, exact=False)
            for variant in variants:
                row = evaluate_variant(question, branches, models, pages, variant)
                results.append(row)
                atomic_json(directory / f"{question['question_id']}_{variant}.json", row)
                values = [row['metrics'][s]['evidence_unit_recall'] for s in ['union', 'rrf20', 'after_top5']]
                complete = row['metrics']['after_top5']['all_required_evidence']
                print(f"  {variant}: 候选证据覆盖 {values[0]:.1%} → RRF前20 {values[1]:.1%} → 重排前5 {values[2]:.1%}；证据齐全={complete}", flush=True)
        summary = summarize(results)
        atomic_json(directory / 'summary.json', {'groups': summary})
        atomic_json(directory / 'run.json', {**provenance, 'status': 'completed'})
        print('\n汇总（平均证据覆盖率 / 证据齐全问题比例）：')
        for item in summary:
            stage = item['stages']['after_top5']
            print(f"{item['variant']} | {item['scope']} | {item['questions']}题 | {stage['mean_evidence_unit_recall']:.1%} / {stage['all_required_evidence_rate']:.1%}")
        print(f"结果：{directory / 'summary.json'}")
    except Exception:
        atomic_json(directory / 'run.json', {**provenance, 'status': 'failed_partial_results_retained'})
        raise
    finally:
        store.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

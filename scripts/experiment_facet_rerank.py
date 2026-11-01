"""Rerank frozen decomposition candidates using cached subquestions.

Default is a preflight preview. --run --question-id paper005 runs one real local
test; --run --limit 0 evaluates the complete source cohort. No retrieval/API.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import RetrievalSettings
from rag.embeddings import LocalModels
from rag.facet_reranking import fuse_facet_rankings, order_for_facet_coverage, facet_proxy, validate_rankings
from rag.generation import select_context
from rag.query_decomposition import QueryPlan
from rag.storage import atomic_json
from scripts.run_retrieval_experiment import rerank_candidates

POLICIES = ['original_rerank', 'facet_rrf', 'facet_cover']
STAGES = ['candidate20', 'after_top5', 'context_budget']


def latest_source(root):
    for directory in sorted((root/'runtime/experiments').glob('decomposition-*'), reverse=True):
        path = directory/'run.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'completed':
            return directory
    raise ValueError('没有已完成的查询拆解实验，请用 --source 指定结果目录。')


def load_cases(source, state, questions, question_ids):
    cases = []
    for qid in question_ids:
        row = json.loads((source/f'{qid}_decomposed.json').read_text(encoding='utf-8'))
        q = questions[qid]
        if row['question_id'] != qid or row['question'] != q['question'] or not q['answerable']:
            raise ValueError('Source row does not match the development question')
        if row['planner_cache']['question_en'] != q['question_en']:
            raise ValueError('Source plan translation differs from the benchmark')
        candidates, original = row['rankings']['rrf20'], row['reranked_candidates']
        if len(candidates) > 20:
            raise ValueError('Expected at most 20 frozen candidates')
        validate_rankings({'original': original, 'candidates': candidates})
        expected = {h['chunk_id']: h for h in candidates}
        for hit in original:
            same = expected[hit['chunk_id']]
            if any(hit.get(field) != same.get(field) for field in ['paper_id', 'title', 'pdf_page', 'document_sha256']):
                raise ValueError('Cached reranking changed source metadata')
            if hashlib.sha256(hit['text'].encode()).hexdigest() != hit['text_sha256']:
                raise ValueError('Candidate text hash differs')
        plan = QueryPlan.model_validate(row['planner_cache']['plan'])
        cases.append({'question': q, 'source_row': row, 'candidates': candidates,
                      'original': original, 'plan': plan,
                      'file_sha256': hashlib.sha256((source/f'{qid}_decomposed.json').read_bytes()).hexdigest()})
    return cases


def make_rankings(case, models):
    rankings, timings = {'original': case['original']}, {}
    for index, sub in enumerate(case['plan'].subqueries, 1):
        label = f'facet{index}'
        start = perf_counter()
        rankings[label] = rerank_candidates(models, sub.zh, case['candidates'], title=True)
        timings[label] = round((perf_counter()-start)*1000, 2)
    validate_rankings(rankings)
    return rankings, timings


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['policy'], 'all_answerable')].append(row)
        groups[(row['policy'], row['scope'])].append(row)
    return [{'policy': policy, 'scope': scope, 'questions': len(items), 'stages': {
        stage: {'evidence_coverage': sum(r['metrics'][stage]['evidence_unit_recall'] for r in items)/len(items),
                'complete_evidence_rate': sum(r['metrics'][stage]['all_required_evidence'] for r in items)/len(items)}
        for stage in STAGES}, 'mean_context_chars': sum(r['context_chars'] for r in items)/len(items)}
        for (policy, scope), items in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Completed decomposition experiment directory')
    parser.add_argument('--run', action='store_true', help='Execute local facet reranking')
    parser.add_argument('--limit', type=int, default=1, help='0=all questions from the source cohort')
    parser.add_argument('--question-id')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    source = (args.source or latest_source(ROOT)).resolve()
    source_path = source/'run.json'
    source_state = json.loads(source_path.read_text(encoding='utf-8'))
    if source_state['status'] != 'completed' or source_state['split'] != 'dev':
        raise ValueError('Use a completed development decomposition experiment')
    controls = source_state['controls']
    if controls['rerank_k'] != 20 or controls['top_k'] != 5 or controls['rerank_input'] != 'title + body':
        raise ValueError('Source reranking controls are incompatible')
    budget = controls['context_char_budget']
    cfg = RetrievalSettings(**source_state['settings'])
    if cfg.chunk_strategy != 'token-window':
        raise ValueError('Keep fixed-window chunking for this experiment')
    benchmark_hash = hashlib.sha256((ROOT/'data/evaluation/paper_benchmark/dev.jsonl').read_bytes()).hexdigest()
    if source_state['benchmark_sha256'] != benchmark_hash:
        raise ValueError('Benchmark has changed since the source experiment')
    questions = {q['question_id']: q for q in load_questions('dev')}
    question_ids = source_state['question_ids']
    if args.question_id:
        if args.question_id not in question_ids:
            parser.error('question-id must occur in the source experiment')
        question_ids = [args.question_id]
    elif args.limit:
        question_ids = question_ids[:args.limit]
    cases = load_cases(source, source_state, questions, question_ids)
    target = ROOT/'runtime/experiments'/('facet-rerank-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    state = {'status': 'preview', 'source_directory': str(source),
             'source_run_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
             'source_row_sha256': {c['question']['question_id']: c['file_sha256'] for c in cases},
             'snapshot_id': source_state['snapshot_id'], 'settings': cfg.model_dump(), 'split': 'dev',
             'benchmark_sha256': benchmark_hash, 'question_ids': question_ids,
             'controls': {'frozen_candidate_limit': 20, 'top_k': 5, 'context_char_budget': budget,
                          'rerank_input': 'title + body', 'facet_query_language': 'zh', 'rrf_k': 60,
                          'original_rank_weight': .5, 'total_facet_rank_weight': .5,
                          'coverage_proxy': 'membership in a facet top3', 'coverage_original_seed': 'original top1'},
             'notes': 'original_rerank reuses the original-question scores on DECOMPOSED candidates, not the earlier single-query retrieval baseline. Extra local query-passage pairs are n_candidates*n_facets. Facet RRF and facet cover reuse the same extra predictions. Original scores are cached; timings including first model load are diagnostic. Rank proxies are not verified evidence support. Gold annotations are used only in scoring, never ordering.'}
    atomic_json(target/'run.json', state)
    for case in cases:
        print(case['question']['question_id'], f"candidates={len(case['candidates'])}",
              f"facets={len(case['plan'].subqueries)}",
              f"additional_pairs={len(case['candidates'])*len(case['plan'].subqueries)}", flush=True)
    if not args.run:
        print('Preview only; add --run. Source:', source)
        print('Result:', target/'run.json')
        return
    try:
        models = LocalModels(cfg)
        pages = load_source_pages({pid for c in cases for pid in c['question']['gold_paper_ids']})
        state['status'] = 'running'
        atomic_json(target/'run.json', state)
        rows, total_pairs = [], 0
        for number, case in enumerate(cases, 1):
            q, cached = case['question'], case['source_row']
            qid = q['question_id']
            rankings, timings = make_rankings(case, models)
            facets = {name: hits for name, hits in rankings.items() if name != 'original'}
            fused = fuse_facet_rankings(rankings)
            covered, steps = order_for_facet_coverage(rankings, fused)
            candidate_metrics = score_evidence(q, case['candidates'], pages, k=max(1, len(case['candidates'])))
            if candidate_metrics['evidence_unit_recall'] != cached['metrics']['rrf20']['evidence_unit_recall']:
                raise ValueError('Frozen candidate scoring differs from the source experiment')
            extra_pairs = len(case['candidates'])*len(facets)
            total_pairs += extra_pairs
            atomic_json(target/f'{qid}_facet_scores.json', {'question_id': qid, 'plan': case['plan'].model_dump(),
                        'rankings': rankings, 'additional_pairs': extra_pairs, 'facet_rerank_ms_including_first_load': timings})
            for policy, ranked in [('original_rerank', rankings['original']), ('facet_rrf', fused), ('facet_cover', covered)]:
                context, used = select_context(ranked, budget)
                metrics = {'candidate20': candidate_metrics,
                           'after_top5': score_evidence(q, ranked[:5], pages, k=5),
                           'context_budget': score_evidence(q, context, pages, k=max(1, len(context)))}
                row = {'question_id': qid, 'question': q['question'], 'scope': q['source_scope'],
                       'policy': policy, 'route': cached['route'], 'metrics': metrics,
                       'reranked_candidates': ranked, 'after_top5': ranked[:5], 'budget_context': context,
                       'context_chars': used, 'candidate_ids': [h['chunk_id'] for h in case['candidates']],
                       'facet_rank_proxy_top5': facet_proxy(facets, ranked[:5]),
                       'facet_rank_proxy_budget': facet_proxy(facets, context),
                       'selection_steps': steps if policy == 'facet_cover' else []}
                atomic_json(target/f'{qid}_{policy}.json', row)
                rows.append({k: row[k] for k in ['question_id', 'policy', 'scope', 'metrics', 'context_chars']})
                print(f'[{number}/{len(cases)}] {qid} {policy}: '+ ' -> '.join(
                    f"{metrics[s]['evidence_unit_recall']:.1%}" for s in STAGES)+f'; context={used}/{budget}', flush=True)
        pairs = defaultdict(dict)
        for row in rows:
            pairs[row['question_id']][row['policy']] = row
        changes = {policy: [{'question_id': qid, 'delta_vs_original_rerank': {
            stage: pair[policy]['metrics'][stage]['evidence_unit_recall']-pair['original_rerank']['metrics'][stage]['evidence_unit_recall']
            for stage in STAGES}} for qid, pair in pairs.items()] for policy in POLICIES[1:]}
        summary = {'groups': summarize(rows), 'paired_changes': changes, 'additional_model_pairs': total_pairs,
                   'improvements_budget': {p: [r['question_id'] for r in items if r['delta_vs_original_rerank']['context_budget'] > 1e-9]
                                           for p, items in changes.items()},
                   'regressions_budget': {p: [r['question_id'] for r in items if r['delta_vs_original_rerank']['context_budget'] < -1e-9]
                                          for p, items in changes.items()}}
        atomic_json(target/'summary.json', summary)
        atomic_json(target/'run.json', {**state, 'status': 'completed', 'additional_model_pairs': total_pairs})
        for group in summary['groups']:
            print(group['policy'], group['scope'], ' | '.join(
                f"{s}: {v['evidence_coverage']:.1%}/{v['complete_evidence_rate']:.1%}" for s, v in group['stages'].items()))
        print('Budget improvements:', summary['improvements_budget'])
        print('Budget regressions:', summary['regressions_budget'])
        print('Additional model pairs:', total_pairs)
        print('Result:', target/'summary.json')
    except Exception:
        atomic_json(target/'run.json', {**state, 'status': 'failed_partial_results_retained'})
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

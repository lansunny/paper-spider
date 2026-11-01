"""Budget-controlled semantic query decomposition on the routed fixed index.

Default previews cached plans only. --prepare makes one DeepSeek request per
uncached question. --run requires prepared plans and runs local retrieval only.
PyCharm: --prepare --limit 1, review the plan, then --run --limit 1.
"""
import argparse
import copy
import hashlib
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.embeddings import LocalModels
from rag.generation import generation_settings, select_context
from rag.ingestion import current_chunks, current_manifest
from rag.query_decomposition import load_or_create_plan, allocate_slots, PROMPT_VERSION
from rag.retrieval import LexicalIndex, reciprocal_rank_fusion
from rag.routing import PaperRouter
from rag.storage import atomic_json
from rag.vector_store import QdrantStore
from scripts.experiment_routing import retrieve_branches, summarize, STAGES
from scripts.run_retrieval_experiment import evaluate_variant


def retrieve_multiquery(lexical, store, models, snapshot_id, question, decision, plan, original_branches):
    queries = [{'zh': question['question'], 'en': question['question_en']},
               *[q.model_dump() for q in plan.subqueries]]
    slots = allocate_slots(100, len(queries))
    allowed = decision['paper_ids'] if decision['mode'] == 'single_paper' else None
    if decision['mode'] == 'single_paper' and len(allowed) != 1:
        raise ValueError('Expected exactly one inherited paper ID')
    histories, rankings = [], {'bm25_en': {}, 'dense': {}}
    for index, (query, top_k) in enumerate(zip(queries, slots, strict=True)):
        label = 'original' if index == 0 else f'facet{index}'
        if index == 0:
            branches = {name: hits[:top_k] for name, hits in original_branches.items()}
        else:
            vector = models.query_vector(query['zh'])
            branches = {'bm25_en': lexical.search(query['en'], top_k=top_k, paper_ids=allowed),
                        'dense': store.search(vector, snapshot_id, top_k=top_k, paper_ids=allowed, exact=False)}
        for name, hits in branches.items():
            if len(hits) > top_k or (allowed and any(h['paper_id'] not in allowed for h in hits)):
                raise ValueError('Retriever violated query budget or inherited scope')
            rankings[name][label] = hits
        histories.append({'label': label, 'queries': query, 'top_k_per_route': top_k,
                          'rankings': branches, 'original_reused': index == 0})
    if len(queries) == 1:
        merged = original_branches
    else:
        merged = {}
        for name, lists in rankings.items():
            hits = reciprocal_rank_fusion(lists, k=60)
            # Preserve facet provenance when the outer BM25/dense RRF replaces ranks.
            merged[name] = [{**h, 'query_ranks': {name: h['ranks']}} for h in hits[:100]]
    trace = {'queries': queries, 'slots_per_route': slots, 'query_rankings': histories,
             'requested_slots_per_route': sum(slots),
             'returned_slots_per_route': {name: sum(len(hits) for hits in lists.values()) for name, lists in rankings.items()},
             'unique_candidates_per_route': {name: len(hits) for name, hits in merged.items()}}
    return merged, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/retrieval.json')
    parser.add_argument('--prepare', action='store_true', help='Allow uncached question-only DeepSeek planning calls')
    parser.add_argument('--run', action='store_true', help='Run local retrieval and rerank; requires cached plans')
    parser.add_argument('--limit', type=int, default=1, help='0=all answerable dev questions')
    parser.add_argument('--question-id')
    parser.add_argument('--budget', type=int, default=8000)
    args = parser.parse_args()
    if args.limit < 0 or args.budget <= 0:
        parser.error('limit must be nonnegative and budget positive')
    cfg = load_settings(args.config)
    manifest = current_manifest(cfg)
    if cfg.corpus != 'papers' or not manifest or manifest['profile'] != cfg.index_profile():
        raise ValueError('Choose an existing compatible papers index')
    if cfg.chunk_strategy != 'token-window':
        raise ValueError('Keep fixed-window chunking for this controlled experiment')
    questions = [q for q in load_questions('dev') if q['answerable']]
    if args.question_id:
        questions = [q for q in questions if q['question_id'] == args.question_id]
        if not questions:
            parser.error('question-id must be an answerable development question')
    elif args.limit:
        questions = questions[:args.limit]
    router, planner = PaperRouter.from_project(manifest['documents']), generation_settings()
    target = ROOT/'runtime/experiments'/('decomposition-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    state = {'status': 'initializing', 'snapshot_id': manifest['snapshot_id'], 'settings': cfg.model_dump(),
             'question_ids': [q['question_id'] for q in questions], 'split': 'dev',
             'benchmark_sha256': hashlib.sha256((ROOT/'data/evaluation/paper_benchmark/dev.jsonl').read_bytes()).hexdigest(),
             'catalog_fingerprint': router.fingerprint, 'planner_model': planner.model, 'prompt_version': PROMPT_VERSION,
             'controls': {'original_route_inherited_by_all_queries': True, 'candidate_slots_per_route': 100,
                          'original_fraction_when_decomposed': 0.5, 'max_facets': 3, 'rrf_k': 60,
                          'rerank_k': 20, 'rerank_query': 'original Chinese', 'rerank_input': 'title + body',
                          'top_k': 5, 'context_char_budget': args.budget, 'gold_used_in_planner_or_retrieval': False},
             'notes': 'Same maximum returned retrieval slots (100 BM25 + 100 dense), not necessarily same unique candidates or latency. Decomposition uses more query embeddings/searches and one cached planning call. Per-route multi-query RRF precedes the existing BM25/dense RRF. English subquery wording is also generated. Character ceilings are not token budgets. Evidence-anchor scores are not answer accuracy.'}
    atomic_json(target/'run.json', state)
    store = None
    try:
        plans, decisions, cache_records, missing = {}, {}, {}, []
        for q in questions:
            qid = q['question_id']
            # The planner sees only these two strings, never evidence or gold labels.
            plan, record = load_or_create_plan(q['question'], q['question_en'], planner, router,
                                               create=args.prepare, root=ROOT)
            decisions[qid], cache_records[qid] = router.route(q['question']), record
            if plan is None:
                missing.append(qid)
                print(qid, 'plan missing; use --prepare', flush=True)
            else:
                plans[qid] = plan
                print(qid, decisions[qid]['mode'], 'cached' if record['cache_hit'] else 'created', flush=True)
                for number, sub in enumerate(plan.subqueries, 1):
                    print(f'  Q{number}: {sub.zh}\n      EN: {sub.en}', flush=True)
                if not plan.subqueries:
                    print('  No decomposition; the original result will be reused.', flush=True)
        state.update(plans=cache_records, decisions=decisions, missing_plans=missing,
                     status='plans_missing' if missing else 'plans_ready')
        atomic_json(target/'run.json', state)
        if not args.run:
            print('Plan preview:', target/'run.json')
            return
        if missing:
            raise RuntimeError('缺少拆解缓存。先用 --prepare 生成并查看，再用 --run 开始检索实验。')
        actual, chunks = current_chunks(cfg)
        if actual['snapshot_id'] != manifest['snapshot_id']:
            raise ValueError('Index changed while preparing plans')
        store = QdrantStore(cfg)
        store.validate_collection()
        if store.count(manifest['snapshot_id']) != len(chunks):
            raise ValueError('Vector count differs from local snapshot')
        lexical, models = LexicalIndex(chunks), LocalModels(cfg)
        pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
        if any(manifest['documents'].get(pid) != digest for pid, digest, _ in pages):
            raise ValueError('Benchmark source differs from indexed PDF')
        state['status'] = 'running'
        atomic_json(target/'run.json', state)
        rows = []
        for number, q in enumerate(questions, 1):
            qid, decision = q['question_id'], decisions[q['question_id']]
            start = perf_counter()
            vector = models.query_vector(q['question'])
            original = retrieve_branches(lexical, store, vector, manifest['snapshot_id'], q, decision)
            original_ms = (perf_counter()-start)*1000
            baseline = evaluate_variant(q, original, models, pages, 'D')
            start = perf_counter()
            branches, trace = retrieve_multiquery(lexical, store, models, manifest['snapshot_id'],
                                                 q, decision, plans[qid], original)
            extra_ms = (perf_counter()-start)*1000
            decomposed = evaluate_variant(q, branches, models, pages, 'D') if plans[qid].subqueries else copy.deepcopy(baseline)
            for policy, row in [('single_query', baseline), ('decomposed', decomposed)]:
                context, used = select_context(row['reranked_candidates'], args.budget)
                row.update(policy=policy, route=decision, budget_context=context, context_chars=used,
                           planner_cache=cache_records[qid], original_retrieval_ms_including_first_load=round(original_ms, 2))
                if policy == 'decomposed':
                    row.update(decomposition=trace, extra_subquery_retrieval_ms=round(extra_ms, 2),
                               reused_single_query_result=not plans[qid].subqueries)
                row['metrics']['context_budget'] = score_evidence(q, context, pages, k=max(1, len(context)))
                atomic_json(target/f'{qid}_{policy}.json', row)
                rows.append({k: row[k] for k in ['question_id', 'policy', 'scope', 'metrics']})
                print(f'[{number}/{len(questions)}] {qid} {policy}: '+ ' -> '.join(
                    f"{row['metrics'][s]['evidence_unit_recall']:.1%}" for s in STAGES), flush=True)
        pairs = defaultdict(dict)
        for row in rows:
            pairs[row['question_id']][row['policy']] = row
        deltas = [{'question_id': qid, 'delta_decomposed_minus_single': {
            stage: pair['decomposed']['metrics'][stage]['evidence_unit_recall']-pair['single_query']['metrics'][stage]['evidence_unit_recall']
            for stage in STAGES}} for qid, pair in pairs.items()]
        report = {'groups': summarize(rows), 'paired_questions': deltas,
                  'regressions_budget': [d['question_id'] for d in deltas if d['delta_decomposed_minus_single']['context_budget'] < -1e-9],
                  'improvements_budget': [d['question_id'] for d in deltas if d['delta_decomposed_minus_single']['context_budget'] > 1e-9]}
        atomic_json(target/'summary.json', report)
        atomic_json(target/'run.json', {**state, 'status': 'completed'})
        for group in report['groups']:
            print(group['policy'], group['scope'], ' | '.join(
                f"{s}: {v['evidence_coverage']:.1%}/{v['complete_evidence_rate']:.1%}" for s, v in group['stages'].items()))
        print('Budget improvements:', report['improvements_budget'])
        print('Budget regressions:', report['regressions_budget'])
        print('Result:', target/'summary.json')
    except Exception:
        atomic_json(target/'run.json', {**state, 'status': 'failed_partial_results_retained'})
        raise
    finally:
        if store:
            store.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

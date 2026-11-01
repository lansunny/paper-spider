"""Compare global D retrieval against question-only single-paper scope routing.

Default: preview one route without loading models. --limit 0 previews all dev
questions. --run performs real local retrieval/reranking (no generation API).
"""
import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_questions, load_source_pages, score_evidence
from rag.config import load_settings
from rag.embeddings import LocalModels
from rag.generation import select_context
from rag.ingestion import current_chunks, current_manifest
from rag.retrieval import LexicalIndex
from rag.routing import PaperRouter, VERSION
from rag.storage import atomic_json
from rag.vector_store import QdrantStore
from scripts.run_retrieval_experiment import evaluate_variant

STAGES = ['union', 'rrf20', 'after_top5', 'context_budget']


def retrieve_branches(lexical, store, vector, snapshot_id, question, decision):
    allowed = decision['paper_ids'] if decision['mode'] == 'single_paper' else None
    if decision['mode'] == 'single_paper' and len(allowed) != 1:
        raise ValueError('A single-paper route must resolve exactly one indexed paper')
    branches = {'bm25_en': lexical.search(question['question_en'], top_k=100, paper_ids=allowed),
                'dense': store.search(vector, snapshot_id, top_k=100, paper_ids=allowed, exact=False)}
    if allowed and any(h['paper_id'] not in allowed for hits in branches.values() for h in hits):
        raise ValueError('Retriever returned a paper outside the routed scope')
    return branches


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['policy'], 'all_answerable')].append(row)
        groups[(row['policy'], row['scope'])].append(row)
    return [{'policy': policy, 'scope': scope, 'questions': len(items), 'stages': {
        stage: {'evidence_coverage': sum(r['metrics'][stage]['evidence_unit_recall'] for r in items)/len(items),
                'complete_evidence_rate': sum(r['metrics'][stage]['all_required_evidence'] for r in items)/len(items)}
        for stage in STAGES}} for (policy, scope), items in sorted(groups.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/retrieval.json')
    parser.add_argument('--run', action='store_true', help='Run actual local retrieval and reranking')
    parser.add_argument('--limit', type=int, default=1, help='0=all answerable dev questions')
    parser.add_argument('--budget', type=int, default=8000)
    args = parser.parse_args()
    if args.limit < 0 or args.budget <= 0:
        parser.error('limit must be nonnegative and budget positive')
    cfg = load_settings(args.config)
    manifest = current_manifest(cfg)
    if cfg.corpus != 'papers' or not manifest or manifest['profile'] != cfg.index_profile():
        raise ValueError('Choose a compatible, built papers index')
    questions = [q for q in load_questions('dev') if q['answerable']]
    if args.limit:
        questions = questions[:args.limit]
    router = PaperRouter.from_project(manifest['documents'])
    # Only user-visible question text is supplied to the router, never gold IDs.
    decisions = {q['question_id']: router.route(q['question']) for q in questions}
    for q in questions:
        d = decisions[q['question_id']]
        print(q['question_id'], d['mode'], d['paper_ids'], d['reason'], flush=True)
    target = ROOT/'runtime/experiments'/('routing-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    state = {'status': 'preview', 'settings': cfg.model_dump(), 'snapshot_id': manifest['snapshot_id'],
             'router_version': VERSION, 'catalog_fingerprint': router.fingerprint,
             'benchmark_sha256': hashlib.sha256((ROOT/'data/evaluation/paper_benchmark/dev.jsonl').read_bytes()).hexdigest(),
             'question_ids': list(decisions), 'decisions': decisions,
             'controls': {'variant': 'D', 'candidate_k_per_route': 100, 'rrf_k': 60, 'rerank_k': 20,
                          'rerank_input': 'title + body', 'top_k': 5, 'context_char_budget': args.budget,
                          'bm25_query': 'fixed English translation', 'dense_rerank_query': 'original Chinese',
                          'filter_source': 'question + catalog only', 'gold_paper_filter': False},
             'notes': 'Same index/models/queries/budgets; only retrieval scope changes. BM25 retains corpus-wide IDF. Global route reuses the unfiltered result exactly. Scores measure labeled evidence, not answer accuracy.'}
    atomic_json(target/'run.json', state)
    if not args.run:
        print('Preview only. Add --run to execute local retrieval. Result:', target/'run.json')
        return
    store, rows = None, []
    try:
        actual, chunks = current_chunks(cfg)
        if actual['snapshot_id'] != manifest['snapshot_id']:
            raise ValueError('Index changed during routing; rerun the experiment')
        store = QdrantStore(cfg)
        store.validate_collection()
        if store.count(manifest['snapshot_id']) != len(chunks):
            raise ValueError('Vector count does not match local snapshot')
        lexical, models = LexicalIndex(chunks), LocalModels(cfg)
        pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
        if any(manifest['documents'].get(pid) != digest for pid, digest, _ in pages):
            raise ValueError('Benchmark PDF and index differ')
        state['status'] = 'running'
        atomic_json(target/'run.json', state)
        for number, question in enumerate(questions, 1):
            qid = question['question_id']
            vector = models.query_vector(question['question'])
            global_decision = {'mode': 'global', 'paper_ids': [], 'reason': 'control'}
            branches = retrieve_branches(lexical, store, vector, manifest['snapshot_id'], question, global_decision)
            baseline = evaluate_variant(question, branches, models, pages, 'D')
            for policy, decision in [('global', global_decision), ('routed', decisions[qid])]:
                reused = policy == 'global' or decision['mode'] == 'global'
                if reused:
                    row = copy.deepcopy(baseline)
                else:
                    branches = retrieve_branches(lexical, store, vector, manifest['snapshot_id'], question, decision)
                    row = evaluate_variant(question, branches, models, pages, 'D')
                context, used = select_context(row['reranked_candidates'], args.budget)
                row.update(policy=policy, route=decision, budget_context=context, context_chars=used,
                           reused_global_result=policy == 'routed' and reused)
                row['metrics']['context_budget'] = score_evidence(question, context, pages, k=max(1, len(context)))
                atomic_json(target/f'{qid}_{policy}.json', row)
                rows.append({k: row[k] for k in ['question_id', 'policy', 'scope', 'metrics']})
                print(f'[{number}/{len(questions)}] {qid} {policy}: '+ ' -> '.join(
                    f"{row['metrics'][s]['evidence_unit_recall']:.1%}" for s in STAGES), flush=True)
        pairs = defaultdict(dict)
        for row in rows:
            pairs[row['question_id']][row['policy']] = row
        deltas = [{'question_id': qid, 'delta_routed_minus_global': {
            stage: pair['routed']['metrics'][stage]['evidence_unit_recall']-pair['global']['metrics'][stage]['evidence_unit_recall']
            for stage in STAGES}} for qid, pair in pairs.items()]
        report = {'groups': summarize(rows), 'paired_questions': deltas,
                  'regressions_budget': [d['question_id'] for d in deltas if d['delta_routed_minus_global']['context_budget'] < -1e-9]}
        atomic_json(target/'summary.json', report)
        atomic_json(target/'run.json', {**state, 'status': 'completed'})
        for group in report['groups']:
            print(group['policy'], group['scope'], ' | '.join(
                f"{s}: {v['evidence_coverage']:.1%}/{v['complete_evidence_rate']:.1%}" for s, v in group['stages'].items()))
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

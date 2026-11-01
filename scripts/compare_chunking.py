"""Paired fixed-window / structural-index experiment using the same D pipeline.

PyCharm defaults to one question. --limit 0 runs all 19 answerable dev questions.
Local embeddings/reranking only: no generation API and no index rebuild.
"""
import argparse
import hashlib
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
from rag.generation import select_context
from rag.ingestion import current_chunks
from rag.retrieval import LexicalIndex
from rag.storage import atomic_json
from rag.vector_store import QdrantStore
from scripts.run_retrieval_experiment import evaluate_variant

STAGES = ['union', 'rrf20', 'after_top5', 'context_budget']
MODEL_FIELDS = ['embedding_model', 'embedding_revision', 'embedding_path',
                'embedding_max_length', 'vector_size', 'reranker_model',
                'reranker_revision', 'reranker_path', 'reranker_max_length']
RUNTIME_FIELDS = ['device', 'batch_size', 'reranker_batch_size']


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['index_label'], 'all_answerable')].append(row)
        groups[(row['index_label'], row['scope'])].append(row)
    output = []
    for (label, scope), items in sorted(groups.items()):
        warm = [r['rerank_ms_including_first_load'] for r in items if r['question_number'] > 1]
        output.append({'index': label, 'scope': scope, 'questions': len(items),
                       'stages': {stage: {
                           'evidence_coverage': sum(r['metrics'][stage]['evidence_unit_recall'] for r in items) / len(items),
                           'complete_evidence_rate': sum(r['metrics'][stage]['all_required_evidence'] for r in items) / len(items),
                       } for stage in STAGES},
                       'mean_context_chars': sum(r['context_chars'] for r in items) / len(items),
                       'mean_context_items': sum(len(r['budget_context']) for r in items) / len(items),
                       'mean_rerank_ms_excluding_first_question': sum(warm)/len(warm) if warm else None})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-config', type=Path, default=ROOT/'configs/retrieval.json')
    parser.add_argument('--structure-config', type=Path, default=ROOT/'configs/retrieval_structure.json')
    parser.add_argument('--limit', type=int, default=1, help='0=all answerable development questions')
    parser.add_argument('--question-id', help='One answerable development question')
    parser.add_argument('--budget', type=int, default=8000, help='Serialized evidence character ceiling')
    args = parser.parse_args()
    if args.limit < 0 or args.budget <= 0:
        parser.error('limit must be nonnegative and budget positive')
    questions = [q for q in load_questions('dev') if q['answerable']]
    if args.question_id:
        questions = [q for q in questions if q['question_id'] == args.question_id]
        if not questions:
            parser.error('question-id must be answerable and in dev')
    elif args.limit:
        questions = questions[:args.limit]
    cfgs = {'fixed': load_settings(args.fixed_config), 'structure': load_settings(args.structure_config)}
    for field in MODEL_FIELDS:
        if getattr(cfgs['fixed'], field) != getattr(cfgs['structure'], field):
            raise ValueError(f'Uncontrolled model setting differs: {field}')
    if any(c.corpus != 'papers' for c in cfgs.values()):
        raise ValueError('Both indexes must use the papers corpus')
    if cfgs['fixed'].chunk_strategy != 'token-window' or cfgs['structure'].chunk_strategy != 'layout-blocks':
        raise ValueError('Expected token-window and layout-blocks configurations')
    if cfgs['fixed'].collection == cfgs['structure'].collection:
        raise ValueError('Use distinct collections')
    indexes, stores, rows = {}, {}, []
    target = ROOT/'runtime/experiments'/('chunking-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    state = {'status': 'initializing', 'created_at': datetime.now(timezone.utc).isoformat(),
             'question_ids': [q['question_id'] for q in questions], 'split': 'dev',
             'benchmark_sha256': hashlib.sha256((ROOT/'data/evaluation/paper_benchmark/dev.jsonl').read_bytes()).hexdigest(),
             'controls': {'variant': 'D', 'bm25_query': 'fixed English translation', 'dense_query': 'original Chinese',
                          'candidate_k_per_route': 100, 'rrf_k': 60, 'rerank_k': 20,
                          'rerank_input': 'title + body', 'top_k': 5, 'context_char_budget': args.budget,
                          'gold_paper_filter': False, 'dense_exact': False,
                          'shared_model_settings': cfgs['fixed'].model_dump(include=set(MODEL_FIELDS + RUNTIME_FIELDS))},
             'runtime_note': 'Both indexes share one LocalModels instance configured from fixed-config. Index-building batch sizes may differ; query embedding and reranking use the recorded shared settings.',
             'budget_note': 'Select from ALL 20 reranked candidates using the same character ceiling. Actual length may differ. Count includes serialized source metadata, excludes prompt wrapper/output, and is not model tokens.',
             'interpretation': 'Structural strategy is a package change, including boundaries, ordering, overlap and merge rules; not a causal estimate for one parameter. Scores measure labeled evidence anchors, not answer accuracy.',
             'timing_note': 'Models/query embeddings shared. Index order alternates. First paired question excluded from rerank timing averages; these are local diagnostic timings.'}
    atomic_json(target/'run.json', state)
    try:
        for label, cfg in cfgs.items():
            manifest, chunks = current_chunks(cfg)
            if manifest['profile'].get('embedding_input', 'body-v1') != 'body-v1':
                raise ValueError('Keep body-only embeddings fixed for this experiment')
            if label == 'structure' and manifest['documents'] != indexes['fixed']['manifest']['documents']:
                raise ValueError('The two indexes contain different documents or PDF versions')
            store = QdrantStore(cfg)
            stores[label] = store
            store.validate_collection()
            if store.count(manifest['snapshot_id']) != len(chunks):
                raise ValueError(f'Qdrant count differs for {label}')
            indexes[label] = {'manifest': manifest, 'lexical': LexicalIndex(chunks)}
            print(f"{label}: {cfg.collection} | {len(chunks)}块 | {manifest['snapshot_id']}", flush=True)
            del chunks
        state['indexes'] = {label: {'collection': cfgs[label].collection,
                                    'settings': cfgs[label].model_dump(),
                                    'snapshot_id': index['manifest']['snapshot_id'],
                                    'chunks': index['manifest']['chunks'],
                                    'profile': index['manifest']['profile']}
                            for label, index in indexes.items()}
        state['status'] = 'running'
        atomic_json(target/'run.json', state)
        # Gold sources are loaded exclusively for scoring, never for retrieval filters.
        pages = load_source_pages({pid for q in questions for pid in q['gold_paper_ids']})
        for pid, digest, _ in pages:
            if indexes['fixed']['manifest']['documents'].get(pid) != digest:
                raise ValueError(f'Benchmark PDF and index differ: {pid}')
        models = LocalModels(cfgs['fixed'])
        del manifest
        for number, question in enumerate(questions, 1):
            print(f"[{number}/{len(questions)}] {question['question_id']} {question['question']}", flush=True)
            vector = models.query_vector(question['question'])
            order = ['fixed', 'structure'] if number % 2 else ['structure', 'fixed']
            for label in order:
                index = indexes[label]
                start = perf_counter()
                branches = {'bm25_en': index['lexical'].search(question['question_en'], top_k=100),
                            'dense': stores[label].search(vector, index['manifest']['snapshot_id'], top_k=100, exact=False)}
                search_ms = (perf_counter()-start)*1000
                row = evaluate_variant(question, branches, models, pages, 'D')
                context, used = select_context(row['reranked_candidates'], args.budget)
                row.update(index_label=label, question_number=number, context_chars=used, budget_context=context,
                           retrieval_ms_excluding_embedding=round(search_ms, 2))
                row['metrics']['context_budget'] = score_evidence(question, context, pages, k=max(1, len(context)))
                atomic_json(target/f"{question['question_id']}_{label}.json", row)
                # Keep only compact records in memory; full trace is already saved.
                rows.append({k: v for k, v in row.items() if k not in {'rankings', 'reranked_candidates'}})
                values = ' → '.join(f"{row['metrics'][s]['evidence_unit_recall']:.1%}" for s in STAGES)
                print(f"  {label}: 候选→融合20→前5→预算上下文: {values}; 上下文{len(context)}项/{used}字符", flush=True)
        by_question = defaultdict(dict)
        for row in rows:
            by_question[row['question_id']][row['index_label']] = row
        paired = []
        for qid, pair in by_question.items():
            paired.append({'question_id': qid, 'delta_structure_minus_fixed': {
                s: pair['structure']['metrics'][s]['evidence_unit_recall'] - pair['fixed']['metrics'][s]['evidence_unit_recall']
                for s in STAGES}})
        summary = {'groups': aggregate(rows), 'paired_questions': paired,
                   'regressions_top5': [p['question_id'] for p in paired if p['delta_structure_minus_fixed']['after_top5'] < -1e-9],
                   'regressions_budget': [p['question_id'] for p in paired if p['delta_structure_minus_fixed']['context_budget'] < -1e-9]}
        atomic_json(target/'summary.json', summary)
        atomic_json(target/'run.json', {**state, 'status': 'completed'})
        print('\n汇总：平均证据覆盖率 / 证据齐全比例')
        for group in summary['groups']:
            values = ' | '.join(f"{s}={group['stages'][s]['evidence_coverage']:.1%}/{group['stages'][s]['complete_evidence_rate']:.1%}" for s in STAGES)
            print(f"{group['index']} | {group['scope']} | {group['questions']}题 | {values}")
        print('前5退步题：', summary['regressions_top5'])
        print('预算上下文退步题：', summary['regressions_budget'])
        print(f"结果：{target/'summary.json'}")
    except Exception:
        atomic_json(target/'run.json', {**state, 'status': 'failed_partial_results_retained'})
        raise
    finally:
        for store in stores.values():
            store.close()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()

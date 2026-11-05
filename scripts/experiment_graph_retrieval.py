"""Compare V2, graph neighbors and matched random scopes inside the same graph corpus."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.benchmark import load_source_pages
from rag.embeddings import LocalModels
from rag.graph_evaluation import (VARIANTS, covered_units, evaluate_case, run_variant,
    scoped_pipeline, select_cases, summarize)
from rag.ingestion import current_chunks, current_manifest, fingerprint
from rag.paper_graph.retrieval import GraphRetrievalOptions
from rag.paper_graph.store import PaperGraphStore
from rag.pipeline_v2 import load_v2_settings
from rag.query_planning_v2 import PaperRouterV2
from rag.retrieval import rerank_candidates
from rag.storage import atomic_json, read_json
from rag.vector_store import QdrantStore


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run local retrieval; preview is the default')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=['dev', 'test50'], default=['dev', 'test50'])
    parser.add_argument('--dimension', choices=['auto', 'topic', 'method', 'experiment', 'conclusion'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='0=all eligible questions')
    parser.add_argument('--resume', type=Path, help='Resume a compatible experiment directory')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    cfg = load_v2_settings(args.config)
    if cfg.corpus != 'papers' or cfg.rerank_k != 20:
        raise ValueError('Use a papers profile with the fixed 20-candidate rerank budget')
    graph_store = PaperGraphStore(ROOT)
    bundle = graph_store.bundle()
    if not bundle:
        raise ValueError('Build the paper graph before running this experiment')
    _, graph_manifest = bundle
    with closing(graph_store._connection(bundle)) as connection:
        graph_documents = {r['paper_id']: r['document_sha256'] for r in connection.execute('SELECT * FROM papers')}
    if len(graph_documents) != graph_manifest['papers']:
        raise ValueError('Graph paper count mismatch')
    manifest = current_manifest(cfg)
    if not manifest or manifest['profile'] != cfg.index_profile():
        raise ValueError('Retrieval profile does not match the active index')
    if any(manifest['documents'].get(pid) != digest for pid, digest in graph_documents.items()):
        raise ValueError('All graph PDFs must match the retrieval snapshot; do not silently test a smaller corpus')
    paper_ids = sorted(graph_documents)
    dataset_paths = {name: ROOT / 'data/evaluation/paper_benchmark' / (name + '.jsonl') for name in args.datasets}
    datasets = {name: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
                for name, path in dataset_paths.items()}
    cases, excluded = select_cases(datasets, paper_ids)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('No answerable questions have all annotated evidence within this graph corpus')
    options = GraphRetrievalOptions(enabled=True, dimension=args.dimension)
    code_files = ['rag/graph_evaluation.py', 'rag/paper_graph/retrieval.py', 'rag/paper_graph/store.py',
        'rag/pipeline_graph.py', 'rag/pipeline_v2.py', 'rag/query_planning_v2.py',
        'rag/retrieval.py', 'rag/lexical_v2.py', 'rag/benchmark.py', 'rag/generation.py',
        'scripts/experiment_graph_retrieval.py']
    design = {'version': 'fixed-graph-corpus-paired-v1', 'retrieval_settings': cfg.model_dump(),
        'index_snapshot_id': manifest['snapshot_id'], 'graph_build_id': graph_manifest['build_id'],
        'graph_manifest_sha256': hashlib.sha256((bundle[0] / 'manifest.json').read_bytes()).hexdigest(),
        'corpus_source': 'all papers in the current graph, independent of benchmark gold',
        'corpus_paper_ids': paper_ids, 'corpus_documents': graph_documents,
        'bm25_idf_corpus': 'only the fixed graph papers', 'qdrant_scope': 'mandatory graph-paper filter on every search',
        'benchmark_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in dataset_paths.items()},
        'questions': [{'dataset': c['dataset'], 'question_id': c['question']['question_id']} for c in cases],
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
        'variants': list(VARIANTS), 'graph_options': options.model_dump(), 'seed': args.seed,
        'controls': {'query': 'original_question_no_gold_translation', 'dense_search': 'exact',
            'candidate_k_per_branch': cfg.candidate_k, 'distinct_rerank_limit': 20, 'top_k': 5,
            'context_char_budget': cfg.context_char_budget, 'source_span_coverage_required': 1.0,
            'rerank_input': 'title + original body', 'generation_api_calls': 0,
            'warmup_excluded': True, 'variant_order': 'seeded balanced rotations',
            'random_control': 'same seed papers, target paper count, per-paper retrieval limits and final candidate budget',
            'rerank_pairs': 'record actual original-plus-facet scoring pairs; not assumed identical'},
        'primary_endpoint': 'macro evidence-unit coverage in budgeted context on non-single-paper routes',
        'single_paper_endpoint': 'no evidence loss or scope widening',
        'uncertainty': 'connected-gold-paper-group bootstrap; fewer than 10 groups is descriptive only',
        'limitations': ['Agent-authored source-anchored labels have not been independently human verified.',
            'Source anchors are not exhaustive relevance labels; unlabelled evidence is not automatically irrelevant.',
            'Existing benchmarks were used previously; this is a scoped diagnostic/regression experiment.',
            'Questions can share gold papers; do not treat them as independent trials.',
            'No generated answers, refusal quality, universal accuracy or production ANN latency are evaluated.']}
    identity = fingerprint(design)
    target = (args.resume or ROOT / 'runtime/experiments' / ('graph-retrieval-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])).resolve()
    if not target.is_relative_to((ROOT / 'runtime/experiments').resolve()):
        raise ValueError('Experiment artifacts must remain under runtime/experiments')
    prior = read_json(target / 'run.json')
    if prior and prior.get('design_sha256') != identity:
        raise ValueError('Resume design differs from the current code, corpus, graph or labels')
    router = PaperRouterV2.from_project(paper_ids, ROOT)
    routes = Counter(router.route(c['question']['question'])['mode'] for c in cases)
    state = {'status': 'prepared', 'design_sha256': identity, 'design': design, 'excluded': excluded,
             'cohorts': dict(Counter(c['dataset'] for c in cases)), 'routes': dict(routes)}
    atomic_json(target / 'run.json', state)
    print(f"Fixed corpus: {len(paper_ids)} graph papers; {len(cases)} answerable questions; "
          f"routes={dict(routes)}; excluded={len(excluded)}", flush=True)
    print('Output:', target, flush=True)
    if not args.run:
        print('Preview only. Add --run for the three-arm local experiment.', flush=True)
        return 0
    full_manifest, chunks = current_chunks(cfg)
    if full_manifest['snapshot_id'] != manifest['snapshot_id']:
        raise ValueError('Index changed during experiment preparation')
    models, base_store = LocalModels(cfg), QdrantStore(cfg)
    pipeline, lexical = scoped_pipeline(cfg, ROOT, models, base_store, manifest, chunks, paper_ids)
    evaluation_papers = {s['paper_id'] for c in cases for u in c['question']['evidence_units']
                         for a in u['alternatives'] for s in a}
    pages = load_source_pages(evaluation_papers, ROOT)
    rng = random.Random(args.seed)
    execution = list(cases)
    rng.shuffle(execution)
    rotation = list(VARIANTS)
    rng.shuffle(rotation)
    rows = []
    try:
        base_store.validate_collection()
        warm = perf_counter()
        models.query_vector('Research paper retrieval evaluation')
        rerank_candidates(models, 'Research paper retrieval evaluation', [lexical.chunks[0].payload()], title=True)
        state.update(status='running', scoped_chunks=len(lexical.chunks), warmup_ms=round((perf_counter()-warm)*1000, 2),
            dependencies={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'sentence-transformers', 'qdrant-client')})
        atomic_json(target / 'run.json', state)
        for index, case in enumerate(execution):
            question = case['question']
            row_path = target / 'cases' / (question['question_id'] + '.json')
            cached = read_json(row_path)
            if cached:
                if cached['design_sha256'] != identity or cached['question'] != question['question']:
                    raise ValueError('Cached case no longer matches the experiment')
                rows.append(cached)
                continue
            if (current_manifest(cfg)['snapshot_id'] != manifest['snapshot_id']
                    or graph_store.summary()['build_id'] != graph_manifest['build_id']):
                raise ValueError('Index or graph changed during the experiment')
            order = [rotation[(index+j) % len(rotation)] for j in range(len(rotation))]
            results, elapsed = {}, {}
            for variant in order:
                start = perf_counter()
                result = run_variant(pipeline, question['question'], variant, options, args.seed, exact=True)
                elapsed[variant] = round((perf_counter()-start)*1000, 2)
                if result.get('graph_retrieval', {}).get('status') == 'fallback':
                    raise ValueError('Graph branch failed; do not treat fallback as a valid graph experiment')
                results[variant] = evaluate_case(question, result, pages, budget=cfg.context_char_budget,
                                                 corpus_paper_ids=set(paper_ids))
            baseline = results['baseline']
            if baseline['route']['mode'] == 'single_paper':
                ids = [h['chunk_id'] for h in baseline['rankings']['context_budget']]
                for variant in ('graph', 'random_scope'):
                    if [h['chunk_id'] for h in results[variant]['rankings']['context_budget']] != ids:
                        raise ValueError('Graph widened or changed a protected single-paper result')
            base_units = covered_units(baseline['metrics']['context_budget'])
            graph_units = covered_units(results['graph']['metrics']['context_budget'])
            row = {'question_id': question['question_id'], 'question': question['question'], 'dataset': case['dataset'],
                'design_sha256': identity, 'variant_order': order, 'elapsed_ms': elapsed, 'results': results,
                'diagnostics': {'gained_units': sorted(graph_units-base_units), 'lost_units': sorted(base_units-graph_units)}}
            atomic_json(row_path, row)
            rows.append(row)
            values = [results[v]['metrics']['context_budget']['evidence_unit_recall'] for v in VARIANTS]
            print(f"[{index+1}/{len(execution)}] {question['question_id']} "
                  f"baseline={values[0]:.1%} graph={values[1]:.1%} random={values[2]:.1%} "
                  f"graph={results['graph']['graph_retrieval'].get('status')}", flush=True)
            state['completed_questions'] = len(rows)
            atomic_json(target / 'run.json', state)
        summary = summarize(rows, cases, seed=args.seed)
        changes = [{'question_id': r['question_id'], 'question': r['question'], **r['diagnostics']}
                   for r in rows if r['diagnostics']['gained_units'] or r['diagnostics']['lost_units']]
        atomic_json(target / 'summary.json', {'design_sha256': identity, 'corpus_papers': len(paper_ids),
            'questions': len(rows), 'generation_api_calls': 0, 'groups': summary, 'changed_cases': changes})
        state.update(status='completed', completed_questions=len(rows))
        atomic_json(target / 'run.json', state)
        for scope in ('expandable_queries_primary', 'single_paper_guard', 'all_scoped_questions'):
            if scope not in summary:
                continue
            print('\n' + scope, flush=True)
            for variant, values in summary[scope]['variants'].items():
                metric = values['stages']['context_budget']
                print(f"  {variant}: coverage={metric['evidence_coverage']:.1%}, "
                      f"complete={metric['complete_evidence_rate']:.1%}, "
                      f"median={values['elapsed_median_ms']:.0f}ms", flush=True)
        print('Summary:', target / 'summary.json', flush=True)
    except BaseException as error:
        state.update(status='failed', completed_questions=len(rows), error_type=type(error).__name__, error=str(error)[:1000])
        atomic_json(target / 'run.json', state)
        raise
    finally:
        base_store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

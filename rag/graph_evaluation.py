"""Paired graph retrieval evaluation within one immutable paper universe."""
from collections import defaultdict
from contextlib import closing
import random

import numpy as np

from rag.benchmark import score_evidence
from rag.generation import select_context
from rag.ingestion import fingerprint
from rag.lexical_v2 import LexicalIndexV2
from rag.paper_graph.retrieval import augment_candidates, plan_expansion
from rag.pipeline_graph import GraphAugmentedPipeline
from rag.pipeline_v2 import PipelineV2
from rag.query_planning_v2 import PaperRouterV2

VARIANTS = ('baseline', 'graph', 'random_scope')
STAGES = ('candidate20', 'top5', 'context_budget')


def select_cases(datasets, paper_ids):
    """Filter a fixed benchmark to this corpus, never filter retrieval by gold IDs."""
    allowed = set(paper_ids)
    selected, excluded, seen = [], [], set()
    for name, questions in datasets.items():
        for question in questions:
            if question['question_id'] in seen:
                raise ValueError('Question IDs overlap between evaluation datasets')
            seen.add(question['question_id'])
            gold = {s['paper_id'] for unit in question['evidence_units']
                    for alternative in unit['alternatives'] for s in alternative}
            reason = ('unanswerable_has_no_positive_retrieval_gold' if not question['answerable'] else
                      'no_gold_evidence' if not gold else
                      'evidence_outside_fixed_graph_corpus' if not gold <= allowed else None)
            if reason:
                excluded.append({'dataset': name, 'question_id': question['question_id'], 'reason': reason,
                                 'outside_paper_ids': sorted(gold - allowed)})
            else:
                selected.append({'dataset': name, 'question': question})
    return selected, excluded


class ScopedLexical(LexicalIndexV2):
    def __init__(self, chunks, root, allowed, *, expand):
        self.allowed = frozenset(allowed)
        super().__init__(chunks, root, expand=expand)

    def search(self, question, top_k=30, paper_ids=None):
        scope = self.allowed if not paper_ids else self.allowed.intersection(paper_ids)
        if not scope:
            return []
        result = super().search(question, top_k, sorted(scope))
        if any(h['paper_id'] not in scope for h in result):
            raise ValueError('Lexical search escaped the fixed experiment corpus')
        return result


class ScopedVectorStore:
    def __init__(self, store, allowed, snapshot_id):
        self.store, self.allowed, self.snapshot_id = store, frozenset(allowed), snapshot_id
        self.calls = []

    def validate_collection(self):
        return self.store.validate_collection()

    def search(self, vector, snapshot_id, top_k=30, paper_ids=None, exact=False):
        if snapshot_id != self.snapshot_id:
            raise ValueError('Dense search switched snapshots during the experiment')
        scope = self.allowed if not paper_ids else self.allowed.intersection(paper_ids)
        if not scope:
            return []
        result = self.store.search(vector, snapshot_id, top_k, sorted(scope), exact)
        if any(h['paper_id'] not in scope for h in result):
            raise ValueError('Dense search escaped the fixed experiment corpus')
        self.calls.append({'scope_size': len(scope), 'limit': top_k, 'hits': len(result)})
        return result

    def close(self):
        self.store.close()


def scoped_pipeline(settings, root, models, store, manifest, chunks, paper_ids):
    allowed = set(paper_ids)
    if not allowed <= set(manifest['documents']):
        raise ValueError('Some graph papers are absent from the retrieval snapshot')
    scoped_chunks = [c for c in chunks if c.paper_id in allowed]
    if {c.paper_id for c in scoped_chunks} != allowed:
        raise ValueError('One or more graph papers have no retrievable chunks')
    local = {**manifest, 'documents': {p: manifest['documents'][p] for p in sorted(allowed)},
             'chunks': len(scoped_chunks), 'chunk_ids_hash': fingerprint([c.chunk_id for c in scoped_chunks]),
             'experiment_corpus_paper_ids': sorted(allowed)}
    lexical = ScopedLexical(scoped_chunks, root, allowed, expand=True)
    vector_store = ScopedVectorStore(store, allowed, manifest['snapshot_id'])
    pipeline = GraphAugmentedPipeline(settings, root=root, models=models, store=vector_store)
    pipeline._router = PaperRouterV2.from_project(allowed, root)
    pipeline.snapshot = lambda: (local, lexical)
    return pipeline, lexical


def matched_random_planner(seed):
    def planner(store, candidates, context, options):
        graph = plan_expansion(store, candidates, context, options)
        count = len(graph['expanded_paper_ids'])
        if not count:
            return graph | {'experiment_control': 'random_scope_matched_to_graph'}
        bundle = store.bundle()
        if bundle[1]['build_id'] != graph['graph_build_id']:
            raise ValueError('Graph changed while creating the matched control')
        with closing(store._connection(bundle)) as connection:
            catalog = {r['paper_id']: dict(r) for r in connection.execute('SELECT * FROM papers')}
        seeds = [s['paper_id'] for s in graph['seeds']]
        allowed = set(context['paper_ids']) if context.get('paper_ids') else None
        eligible = sorted(pid for pid, paper in catalog.items() if pid not in seeds
            and context['manifest']['documents'].get(pid) == paper['document_sha256']
            and (allowed is None or pid in allowed))
        identity = fingerprint([seed, context['question'], graph['dimension'], seeds])
        rng = random.Random(int(identity[:16], 16))
        rng.shuffle(eligible)
        targets = eligible[:count]
        paths = [{'source': seeds[i % len(seeds)], 'target': target,
            'source_title': catalog[seeds[i % len(seeds)]]['title'], 'target_title': catalog[target]['title'],
            'dimension': graph['dimension'], 'weight': None,
            'relation': 'random_control_assignment', 'not_a_graph_edge': True}
            for i, target in enumerate(targets)]
        return {**graph, 'expanded_paper_ids': targets, 'paths': paths,
            'experiment_control': 'random_scope_matched_to_graph', 'control_seed': seed,
            'matched_target_paper_count': count,
            'overlap_with_graph_targets': len(set(targets) & set(graph['expanded_paper_ids']))}
    return planner


def run_variant(pipeline, question, variant, options, seed, *, exact=True):
    kwargs = {'mode': 'hybrid', 'top_k': 5, 'rerank': True, 'exact': exact, 'collect_trace': True}
    if variant == 'baseline':
        return pipeline.retrieve(question, **kwargs)
    if variant == 'graph':
        return pipeline.retrieve(question, graph_retrieval=options.model_dump(), **kwargs)
    if variant != 'random_scope':
        raise ValueError('Unknown graph experiment variant')

    def transform(candidates, context):
        hits, extension = augment_candidates(pipeline, candidates, context, options,
            planner=matched_random_planner(seed))
        for hit in hits:
            if hit.get('retrieval_origin') == 'graph_expansion':
                hit['retrieval_origin'] = 'random_scope_expansion'
                hit['control_paths'] = hit.pop('graph_paths', [])
        return hits, extension

    return PipelineV2.retrieve(pipeline, question, candidate_transform=transform, **kwargs)


def evaluate_case(question, result, pages, *, budget, corpus_paper_ids):
    context, chars = select_context(result['reranked_candidates'], budget)
    rankings = {'candidate20': result['rankings']['rrf20'], 'top5': result['reranked_candidates'][:5],
                'context_budget': context}
    if any(h['paper_id'] not in corpus_paper_ids for hits in rankings.values() for h in hits):
        raise ValueError('An evaluated result escaped the fixed paper corpus')
    if len(rankings['candidate20']) > 20 or chars > budget:
        raise ValueError('Experiment candidate/context budget exceeded')
    metrics = {stage: score_evidence(question, hits, pages, k=max(1, len(hits))) for stage, hits in rankings.items()}
    info = result.get('graph_retrieval', {})
    expanded = [h for h in context if h.get('retrieval_origin') in {'graph_expansion', 'random_scope_expansion'}]
    without_expansion = [h for h in context if h not in expanded]
    without = score_evidence(question, without_expansion, pages, k=max(1, len(without_expansion)))
    whole_units = covered_units(metrics['context_budget'])
    return {'metrics': metrics, 'context_chars': chars, 'context_chunks': len(context),
        'scope_guard_passed': True, 'expanded_context_chunk_ids': [h['chunk_id'] for h in expanded],
        'expanded_context_paper_ids': sorted({h['paper_id'] for h in expanded}),
        'units_requiring_expansion_in_this_context': sorted(whole_units - covered_units(without)),
        'graph_retrieval': info, 'route': result['route'], 'candidate_counts': result['candidate_counts'],
        'timings': result['timings'], 'rankings': rankings, 'reranked_candidates': result['reranked_candidates'],
        'query_plan': result['query_plan']}


def covered_units(metric):
    return {u['unit_id'] for u in metric['units'] if u['covered']}


def paper_clusters(cases):
    """Conservatively group questions sharing any gold paper, including cross-paper links."""
    parent = {}
    def find(p):
        parent.setdefault(p, p)
        if parent[p] != p:
            parent[p] = find(parent[p])
        return parent[p]
    groups = {}
    for case in cases:
        q = case['question']
        papers = sorted({s['paper_id'] for u in q['evidence_units'] for a in u['alternatives'] for s in a})
        for p in papers:
            parent[find(p)] = find(papers[0])
        groups[q['question_id']] = papers[0]
    return {qid: find(p) for qid, p in groups.items()}


def clustered_interval(deltas, cluster_ids, seed=42, samples=4000):
    groups = defaultdict(list)
    for delta, cluster in zip(deltas, cluster_ids, strict=True):
        groups[cluster].append(delta)
    count = len(groups)
    if count < 3:
        return {'ci95': None, 'paper_groups': count, 'reliable': False,
                'reason': 'fewer_than_three_independent_paper_groups'}
    sums = np.array([sum(v) for v in groups.values()])
    sizes = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, count, size=(samples, count))
    values = sums[selections].sum(axis=1) / sizes[selections].sum(axis=1)
    return {'ci95': [float(v) for v in np.quantile(values, [.025, .975])],
            'paper_groups': count, 'reliable': count >= 10,
            'reason': 'few_paper_groups_descriptive_only' if count < 10 else 'cluster_bootstrap',
            'resamples': samples, 'unit': 'connected_gold_paper_groups'}


def summarize(rows, cases, *, seed):
    clusters = paper_clusters(cases)
    scopes = {'all_scoped_questions': rows,
        'expandable_queries_primary': [r for r in rows if r['results']['baseline']['route']['mode'] != 'single_paper'],
        'single_paper_guard': [r for r in rows if r['results']['baseline']['route']['mode'] == 'single_paper']}
    for dataset in sorted({r['dataset'] for r in rows}):
        scopes['dataset_' + dataset] = [r for r in rows if r['dataset'] == dataset]
    output = {}
    for name, selected in scopes.items():
        if not selected:
            continue
        aggregate = {}
        for variant in VARIANTS:
            records = [r['results'][variant] for r in selected]
            times = [r['elapsed_ms'][variant] for r in selected]
            aggregate[variant] = {'questions': len(selected), 'stages': {stage: {
                'evidence_coverage': float(np.mean([r['metrics'][stage]['evidence_unit_recall'] for r in records])),
                'complete_evidence_rate': float(np.mean([r['metrics'][stage]['all_required_evidence'] for r in records]))}
                for stage in STAGES}, 'elapsed_median_ms': float(np.median(times)),
                'elapsed_p95_ms': float(np.quantile(times, .95)),
                'mean_rerank_pairs': float(np.mean([r['candidate_counts']['rerank_pairs'] for r in records])),
                'expanded_questions': sum(r['graph_retrieval'].get('status') == 'expanded' for r in records),
                'extra_dense_queries': sum(r['graph_retrieval'].get('extra_dense_queries', 0) for r in records)}
        comparisons = {}
        for a, b in [('graph', 'baseline'), ('random_scope', 'baseline'), ('graph', 'random_scope')]:
            deltas = [r['results'][a]['metrics']['context_budget']['evidence_unit_recall'] -
                      r['results'][b]['metrics']['context_budget']['evidence_unit_recall'] for r in selected]
            comparisons[a + '_minus_' + b] = {'mean_delta': float(np.mean(deltas)),
                'wins': sum(d > 1e-12 for d in deltas), 'losses': sum(d < -1e-12 for d in deltas),
                'ties': sum(abs(d) <= 1e-12 for d in deltas),
                'cluster_uncertainty': clustered_interval(deltas, [clusters[r['question_id']] for r in selected], seed)}
        output[name] = {'variants': aggregate, 'comparisons': comparisons}
    return output

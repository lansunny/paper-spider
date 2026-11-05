"""Bounded one-hop paper expansion; graph summaries never become answer evidence."""
import sqlite3
from contextlib import closing
from time import perf_counter
from typing import Literal

from pydantic import Field
import httpx
from qdrant_client.http.exceptions import ApiException

from rag.retrieval import reciprocal_rank_fusion
from .schema import StrictModel
from .store import PaperGraphStore

VERSION = 'paper-similarity-one-hop-v1'


class GraphRetrievalOptions(StrictModel):
    enabled: bool = False
    dimension: Literal['auto', 'topic', 'method', 'experiment', 'conclusion'] = 'auto'
    seed_limit: int = Field(default=2, ge=1, le=3)
    neighbors_per_seed: int = Field(default=2, ge=1, le=3)
    max_papers: int = Field(default=4, ge=1, le=6)
    candidate_slots: int = Field(default=4, ge=1, le=8)
    candidates_per_paper: int = Field(default=8, ge=1, le=20)


def choose_dimension(question, requested):
    if requested != 'auto':
        return requested, 'user_selected'
    text = question.casefold()
    choices = [('experiment', ('实验设置', '数据集', '基线', '实验结果', 'benchmark', 'dataset', 'baseline', 'hyperparameter')),
               ('conclusion', ('局限', '适用范围', '主要发现', '结论', 'limitations', 'findings', 'conclusion')),
               ('method', ('方法', '机制', '流程', '训练', '推理', '触发', '如何', 'method', 'training', 'reasoning', 'how'))]
    return next(((dimension, 'question_keyword_rule') for dimension, terms in choices if any(t in text for t in terms)),
                ('topic', 'topic_default'))


def plan_expansion(graph_store, candidates, context, options):
    """Pin one graph bundle, checking both endpoints against the active PDF versions."""
    dimension, selection = choose_dimension(context['question'], options.dimension)
    trace = {'enabled': True, 'version': VERSION, 'status': 'skipped', 'dimension': dimension,
        'dimension_selection': selection, 'reason': '', 'graph_build_id': None, 'seeds': [], 'paths': [],
        'expanded_paper_ids': [], 'added_candidate_ids': [], 'max_hops': 1,
        'rerank_budget': context['rerank_total'], 'candidate_slots': min(options.candidate_slots, context['rerank_total']//2),
        'extra_bm25_queries': 0, 'extra_dense_queries': 0, 'summary_used_as_evidence': False}
    if context['route']['mode'] == 'single_paper' or len(context.get('paper_ids') or []) == 1:
        return trace | {'reason': 'single_paper_scope'}
    if trace['candidate_slots'] < 1:
        return trace | {'reason': 'insufficient_candidate_budget'}
    bundle = graph_store.bundle()
    if bundle is None:
        return trace | {'reason': 'graph_not_built'}
    documents = context['manifest']['documents']
    allowed = set(context['paper_ids']) if context.get('paper_ids') else None
    with closing(graph_store._connection(bundle)) as connection:
        catalog = {r['paper_id']: dict(r) for r in connection.execute('SELECT * FROM papers')}
        eligible = {pid for pid, paper in catalog.items() if documents.get(pid) == paper['document_sha256']
                    and (allowed is None or pid in allowed)}
        explicit = context['route'].get('matched_paper_ids', [])
        resolved_explicit = [pid for pid in explicit if pid in eligible]
        ordered = list(dict.fromkeys(resolved_explicit or [h['paper_id'] for h in candidates]))
        seeds = [pid for pid in ordered if pid in eligible][:options.seed_limit]
        trace.update(graph_build_id=bundle[1]['build_id'], graph_papers=len(catalog), eligible_graph_papers=len(eligible),
            graph_source_snapshot_id=bundle[1]['source_snapshot_id'], retrieval_snapshot_id=context['manifest']['snapshot_id'],
            compatibility='document_sha256_not_chunker_or_snapshot_id',
            seed_selection='explicit_papers_only' if resolved_explicit else 'retrieved_candidate_order',
            seeds=[{'paper_id': pid, 'title': catalog[pid]['title'],
                    'source': 'explicit_mention' if pid in explicit else 'retrieved_candidate'} for pid in seeds])
        if not seeds:
            return trace | {'reason': 'no_compatible_seed_in_pilot_graph'}
        paths = []
        for seed in seeds:
            rows = connection.execute('''SELECT source,target,weight FROM edges WHERE dimension=?
                AND (source=? OR target=?) ORDER BY weight DESC,source,target''', (dimension, seed, seed))
            used = 0
            for row in rows:
                target = row['target'] if row['source'] == seed else row['source']
                if target not in eligible or target in seeds:
                    continue
                paths.append({'source': seed, 'target': target, 'dimension': dimension, 'weight': row['weight'],
                    'source_title': catalog[seed]['title'], 'target_title': catalog[target]['title'],
                    'source_profile_sha256': catalog[seed]['profile_sha256'],
                    'target_profile_sha256': catalog[target]['profile_sha256'], 'relation': dimension + '_similarity'})
                used += 1
                if used >= options.neighbors_per_seed:
                    break
        # Fairly visit one neighbor from each seed before taking their next neighbor.
        targets = []
        for rank in range(options.neighbors_per_seed):
            for seed in seeds:
                neighbors = [p['target'] for p in paths if p['source'] == seed]
                if rank < len(neighbors) and neighbors[rank] not in targets:
                    targets.append(neighbors[rank])
        targets = targets[:min(options.max_papers, trace['candidate_slots'])]
        trace.update(paths=[p for p in paths if p['target'] in targets], expanded_paper_ids=targets,
                     reason='neighbors_selected' if targets else 'no_eligible_neighbors')
    return trace


def augment_candidates(pipeline, candidates, context, options, *, planner=None):
    start = perf_counter()
    trace = None
    try:
        trace = (planner or plan_expansion)(PaperGraphStore(pipeline.root), candidates, context, options)
        if not trace['expanded_paper_ids']:
            return candidates, {'graph_retrieval': trace}
        paths = {pid: [p for p in trace['paths'] if p['target'] == pid] for pid in trace['expanded_paper_ids']}
        pools, counts = {}, {}
        vector = context['query_vectors'].get(context['question'])
        for pid in trace['expanded_paper_ids']:
            branches = {}
            if context['mode'] in {'bm25', 'hybrid'}:
                branches['bm25'] = context['lexical'].search(context['lexical_query'], options.candidates_per_paper, [pid])
                trace['extra_bm25_queries'] += 1
            if context['mode'] in {'dense', 'hybrid'}:
                if vector is None:
                    vector = pipeline.models.query_vector(context['question'])
                branches['dense'] = pipeline.store.search(vector, context['manifest']['snapshot_id'],
                    options.candidates_per_paper, [pid], context['exact'])
                trace['extra_dense_queries'] += 1
            for hits in branches.values():
                for hit in hits:
                    if hit['paper_id'] != pid or hit['document_sha256'] != context['manifest']['documents'][pid]:
                        raise ValueError('Graph branch returned a passage outside the selected PDF version/scope')
            pool = reciprocal_rank_fusion(branches, pipeline.settings.rrf_k) if context['mode'] == 'hybrid' else branches[context['mode']]
            pools[pid] = pool
            counts[pid] = {name: len(hits) for name, hits in branches.items()}
        # Only genuinely new chunks consume the reserved graph slots.
        seen = {h['chunk_id'] for h in candidates}
        additions = []
        for offset in range(options.candidates_per_paper * 2):
            for pid, pool in pools.items():
                if offset < len(pool) and pool[offset]['chunk_id'] not in seen:
                    hit = pool[offset]
                    additions.append({**hit, 'retrieval_origin': 'graph_expansion', 'graph_paths': paths[pid]})
                    seen.add(hit['chunk_id'])
                    if len(additions) >= trace['candidate_slots']:
                        break
            if len(additions) >= trace['candidate_slots']:
                break
        if not additions:
            return candidates, {'graph_retrieval': trace | {'reason': 'no_new_body_candidates', 'branch_counts': counts}}
        limit = context['rerank_total']
        baseline_count = min(len(candidates), limit - len(additions))
        # Preserve formula rescue and one representative for explicitly decomposed papers.
        required = set(context['protected_chunk_ids'])
        for facet in context['facets']:
            first = next((h for h in candidates if h['paper_id'] == facet.paper_id), None)
            if first:
                required.add(first['chunk_id'])
        keep = [h for h in candidates if h['chunk_id'] in required]
        if len(keep) > baseline_count:
            additions = additions[:max(0, limit - len(keep))]
            baseline_count = min(len(candidates), limit - len(additions))
        keep_ids = {h['chunk_id'] for h in keep}
        for hit in candidates:
            if len(keep) >= baseline_count:
                break
            if hit['chunk_id'] not in keep_ids:
                keep.append(hit)
                keep_ids.add(hit['chunk_id'])
        # Preserve the original relative ordering of all retained baseline chunks.
        kept = [{**h, 'retrieval_origin': 'baseline'} for h in candidates if h['chunk_id'] in keep_ids]
        chosen = [*kept, *additions][:limit]
        trace.update(status='expanded' if additions else 'skipped', reason='body_candidates_added' if additions else 'protected_baseline_budget',
            branch_counts=counts, added_candidate_ids=[h['chunk_id'] for h in additions],
            baseline_candidate_ids=[h['chunk_id'] for h in candidates],
            displaced_baseline_candidate_ids=[h['chunk_id'] for h in candidates if h['chunk_id'] not in keep_ids],
            expanded_candidates=len(additions), elapsed_ms=round((perf_counter()-start)*1000, 2))
        return chosen, {'graph_retrieval': trace}
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, httpx.HTTPError, ApiException) as error:
        dimension, _ = choose_dimension(context['question'], options.dimension)
        return candidates, {'graph_retrieval': {**(trace or {}), 'enabled': True, 'status': 'fallback',
            'dimension': dimension, 'reason': 'graph_expansion_unavailable', 'error_type': type(error).__name__,
            'added_candidate_ids': [], 'summary_used_as_evidence': False}}

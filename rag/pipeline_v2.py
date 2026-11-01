"""Experimental retrieval V2; frozen V1 indexes, bodies and code stay reusable."""
from dataclasses import asdict, dataclass
from time import perf_counter

from rag.config import ROOT, load_settings
from rag.facet_reranking import fuse_facet_rankings
from rag.ingestion import current_chunks, current_manifest
from rag.lexical_v2 import LexicalIndexV2
from rag.pipeline import Pipeline
from rag.query_decomposition import allocate_slots
from rag.query_planning_v2 import PaperRouterV2, balanced_candidates, paper_facets
from rag.retrieval import reciprocal_rank_fusion, rerank_candidates

VERSION = 'scoped-facets-lexical-context-v2'


@dataclass(frozen=True)
class RetrievalPolicyV2:
    decomposition: bool = True
    lexical_expansion: bool = True
    formula_ranking: bool = True
    formula_rescue: bool = True


def load_v2_settings(path=None, root=ROOT):
    return load_settings(path or root/'configs/retrieval_v2.json')


class PipelineV2(Pipeline):
    def __init__(self, settings, root=ROOT, models=None, store=None, *, policy=None):
        super().__init__(settings, root=root, models=models, store=store)
        self.policy = policy or RetrievalPolicyV2()
        self._formula_store = None
        self._formula_registry_version = None

    def snapshot(self):
        with self.lock:
            latest = current_manifest(self.settings, self.root)
            if latest is None or self._snapshot is None or latest['snapshot_id'] != self._snapshot['snapshot_id']:
                manifest, chunks = current_chunks(self.settings, self.root)
                self._snapshot = manifest
                self._lexical = LexicalIndexV2(chunks, self.root, expand=self.policy.lexical_expansion)
                self._router = (PaperRouterV2.from_project(manifest['documents'], self.root)
                                if self.settings.corpus == 'papers' else None)
            if self._snapshot['profile'] != self.settings.index_profile():
                raise ValueError('索引配置不匹配，请选择已有索引的配置。')
            return self._snapshot, self._lexical

    def retrieve(self, question, mode='hybrid', top_k=5, rerank=True, paper_ids=None, exact=False,
                 *, lexical_query=None, auto_route=True, collect_trace=False):
        if mode not in {'bm25', 'dense', 'hybrid'} or not 1 <= top_k <= 100:
            raise ValueError('Invalid retrieval mode/top_k.')
        question = question.strip()
        if not question:
            raise ValueError('请输入问题。')
        manifest, lexical = self.snapshot()
        if paper_ids and not set(paper_ids).issubset(manifest['documents']):
            raise ValueError('选择的论文尚未进入当前索引。')
        if paper_ids:
            route = {'mode': 'manual', 'paper_ids': list(paper_ids), 'reason': 'explicit_user_selection'}
        elif auto_route and self._router is not None:
            route = self._router.route(question)
            paper_ids = route['paper_ids'] or None
        else:
            route = {'mode': 'global', 'paper_ids': [], 'reason': 'routing_disabled_or_other_corpus'}
        facets = (paper_facets(question, self._router, allowed_ids=paper_ids)
                  if self.policy.decomposition and auto_route and self._router is not None else [])
        candidate_total = max(top_k, self.settings.candidate_k)
        rerank_total = max(top_k, self.settings.rerank_k)
        # Keep branch slots and distinct CE candidates fixed. Facet scoring adds
        # query-candidate pairs; candidate_counts reports that extra local cost.
        if len(facets)+1 > min(candidate_total, rerank_total):
            facets = []
        quotas = allocate_slots(candidate_total, len(facets)+1)
        lexical_text = question if lexical_query is None else lexical_query.strip()
        if not lexical_text:
            raise ValueError('BM25检索句不能为空。')
        queries = [(question, lexical_text, paper_ids), *[(f.question, f.question, [f.paper_id]) for f in facets]]
        pools, per_branch, timings, query_trace = [], {}, {}, []
        if mode in {'dense', 'hybrid'}:
            self.store.validate_collection()
        for i, ((query, lex_query, scope), quota) in enumerate(zip(queries, quotas)):
            label = 'original' if i == 0 else f'paper_{i}'
            branches = {}
            if mode in {'bm25', 'hybrid'}:
                start = perf_counter()
                branches['bm25'] = lexical.search(lex_query, quota, scope)
                timings['bm25_ms'] = timings.get('bm25_ms', 0)+(perf_counter()-start)*1000
            if mode in {'dense', 'hybrid'}:
                start = perf_counter()
                vector = self.models.query_vector(query)
                timings['query_embedding_ms'] = timings.get('query_embedding_ms', 0)+(perf_counter()-start)*1000
                start = perf_counter()
                branches['dense'] = self.store.search(vector, manifest['snapshot_id'], quota, scope, exact)
                timings['dense_search_ms'] = timings.get('dense_search_ms', 0)+(perf_counter()-start)*1000
            if scope and any(h['paper_id'] not in scope for hits in branches.values() for h in hits):
                raise ValueError('子查询结果超出了选定论文范围。')
            pools.append(reciprocal_rank_fusion(branches, self.settings.rrf_k) if mode == 'hybrid' else branches[mode])
            for branch, hits in branches.items():
                per_branch.setdefault(branch, {})[label] = hits
            query_trace.append({'label': label, 'question': query, 'paper_ids': scope or [],
                                'slots_per_branch': quota, 'lexical': lexical.query(lex_query),
                                'branch_counts': {k: len(v) for k, v in branches.items()}})
        aggregated = {name: (next(iter(rankings.values())) if len(rankings) == 1 else
                            reciprocal_rank_fusion(rankings, self.settings.rrf_k))
                      for name, rankings in per_branch.items()}
        fused = (pools[0] if len(pools) == 1 else reciprocal_rank_fusion(
            {f'query_{i}': hits for i, hits in enumerate(pools)}, self.settings.rrf_k))
        candidate_hits = balanced_candidates(pools[0], pools[1:], fused, rerank_total) if facets else fused[:rerank_total]
        before_formula_rescue = candidate_hits
        rescue = {'performed':False,'reason':'disabled_or_no_cross_paper_facets'}
        if facets and rerank and self.policy.formula_ranking and self.policy.formula_rescue:
            from rag.formula_retrieval_v2 import rescue_formula_sources
            from rag.formula_store import FormulaStore
            start=perf_counter()
            try:
                registry=self.root/'runtime/formulas/CURRENT.json'
                stat=registry.stat() if registry.exists() else None
                registry_version=(stat.st_mtime_ns,stat.st_size) if stat else None
                if self._formula_store is None or registry_version!=self._formula_registry_version:
                    self._formula_store=FormulaStore(self.root)
                    self._formula_registry_version=registry_version
                rescued,rescue=rescue_formula_sources(question,fused,[f.record() for f in facets],
                    models=self.models,documents=manifest['documents'],root=self.root,store=self._formula_store)
                by_id={h['chunk_id']:h for h in fused}
                candidate_hits=[*[by_id[cid] for cid in rescued],
                                *[h for h in candidate_hits if h['chunk_id'] not in rescued]][:rerank_total]
            except (ValueError,OSError,KeyError) as error:
                rescue={'performed':False,'reason':'formula_artifact_validation_failed','error_type':type(error).__name__}
            timings['formula_candidate_selection_ms']=(perf_counter()-start)*1000
        rerank_calls = 0
        if rerank:
            start = perf_counter()
            rankings = {'original': rerank_candidates(self.models, question, candidate_hits, title=True)}
            rerank_calls += len(candidate_hits)
            for i, facet in enumerate(facets, 1):
                scoped = [h for h in candidate_hits if h['paper_id'] == facet.paper_id]
                ranked = rerank_candidates(self.models, facet.question, scoped, title=True)
                rankings[f'paper_{i}'] = [*ranked, *[h for h in rankings['original'] if h['paper_id'] != facet.paper_id]]
                rerank_calls += len(scoped)
            ranked = fuse_facet_rankings(rankings, k=self.settings.rrf_k)
            # Ensure each explicitly requested paper has an early representative.
            # This is a ranking heuristic, never proof that its evidence is sufficient.
            front = rankings['original'][:1]
            for i, facet in enumerate(facets, 1):
                best = next((h for h in rankings[f'paper_{i}'] if h['paper_id'] == facet.paper_id), None)
                if best and best['chunk_id'] not in {h['chunk_id'] for h in front}:
                    front.append(best)
            ids = {h['chunk_id'] for h in front}
            by_id = {h['chunk_id']: h for h in ranked}
            ranked = [*[by_id[h['chunk_id']] for h in front], *[h for h in ranked if h['chunk_id'] not in ids]]
            timings['rerank_ms'] = (perf_counter()-start)*1000
        else:
            ranked = candidate_hits
        result = {'evidence': ranked[:top_k], 'reranked_candidates': ranked, 'snapshot_id': manifest['snapshot_id'],
                  'pipeline_version': VERSION, 'policy': asdict(self.policy), 'route': route,
                  'query_plan': {'method': 'explicit_paper_scoped_facets', 'api_called': False,
                                 'facets': [f.record() for f in facets], 'queries': query_trace},
                  'bm25_query': lexical.query(lexical_text)['text'],
                  'bm25_query_source': 'provided_plus_glossary' if lexical_query is not None else 'question_plus_glossary',
                  'rerank_input': 'title + body; original plus paper facets' if rerank else None,
                  'context_policy': {'eligible': 'all_reranked_candidates', 'char_budget': self.settings.context_char_budget},
                  'candidate_selection': {'policy':'per_paper_slots_plus_linked_formula_sources', 'formula_rescue':rescue},
                  'timings': {k: round(v, 2) for k, v in timings.items()},
                  'candidate_counts': {**{k: len(v) for k, v in aggregated.items()}, 'merged': len(fused),
                                       'rerank_candidates': len(candidate_hits), 'rerank_pairs': rerank_calls,
                                       'queries': len(queries), 'slots_per_branch': candidate_total}}
        if collect_trace:
            result['rankings'] = {**aggregated, 'union': fused, 'rrf20_before_formula_rescue':before_formula_rescue,
                                  'rrf20': candidate_hits, 'after_top5': ranked[:top_k]}
        return result

    def attach_formulas(self, question, evidence, **kwargs):
        from rag.formula_retrieval_v2 import attach_formulas_v2
        return attach_formulas_v2(question, evidence, root=self.root, models=self.models,
                                  rank=self.policy.formula_ranking, **kwargs)

"""Opt-in graph augmentation of V2; disabled requests use the exact V2 path."""
from rag.pipeline_v2 import PipelineV2
from rag.paper_graph.retrieval import GraphRetrievalOptions, augment_candidates


class GraphAugmentedPipeline(PipelineV2):
    def retrieve(self, question, mode='hybrid', top_k=5, rerank=True, paper_ids=None, exact=False,
                 *, lexical_query=None, auto_route=True, collect_trace=False, graph_retrieval=None):
        options = GraphRetrievalOptions.model_validate(graph_retrieval or {})
        kwargs = {'lexical_query': lexical_query, 'auto_route': auto_route, 'collect_trace': collect_trace}
        if options.enabled:
            kwargs['candidate_transform'] = lambda candidates, context: augment_candidates(self, candidates, context, options)
        result = super().retrieve(question, mode, top_k, rerank, paper_ids, exact, **kwargs)
        if options.enabled:
            info = result['graph_retrieval']
            result['candidate_counts']['graph_added'] = len(info.get('added_candidate_ids', []))
            result['candidate_counts']['graph_scoped_queries'] = info.get('extra_bm25_queries', 0) + info.get('extra_dense_queries', 0)
            if 'elapsed_ms' in info:
                result['timings']['graph_expansion_ms'] = info['elapsed_ms']
        return result

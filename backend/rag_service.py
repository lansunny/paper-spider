"""Web adapter for the independently testable RAG pipeline."""
from functools import lru_cache
from time import perf_counter
import os
from rag.config import load_settings
from rag.generation import generation_settings
from rag.ingestion import current_manifest
from rag.jobs import IndexJobs
from rag.pipeline import Pipeline
from rag.formula_retrieval import attach_formulas


@lru_cache(maxsize=1)
def get_pipeline():
    profile=os.environ.get('PAPERRAG_RETRIEVAL_PROFILE','v2')
    if profile=='v1':
        return Pipeline(load_settings())
    if profile!='v2':
        raise ValueError('PAPERRAG_RETRIEVAL_PROFILE must be v1 or v2')
    from rag.pipeline_v2 import PipelineV2,load_v2_settings
    return PipelineV2(load_v2_settings())


@lru_cache(maxsize=1)
def get_jobs():
    return IndexJobs(get_pipeline())


def status():
    pipeline = get_pipeline()
    manifest = current_manifest(pipeline.settings, pipeline.root)
    generation = generation_settings(pipeline.root)
    return {'mode': 'rag' if generation.configured else 'retrieval_only',
            'generation_configured': generation.configured, 'model': generation.model,
            'snapshot_id': manifest['snapshot_id'] if manifest else None,
            'indexed_documents': manifest['documents'] if manifest else {},
            'indexed_chunks': manifest['chunks'] if manifest else 0,
            'embedding_ready': (pipeline.root / pipeline.settings.embedding_path / 'paperrag-model.json').exists(),
            'reranker_ready': (pipeline.root / pipeline.settings.reranker_path / 'paperrag-model.json').exists()}


def ask(question, paper_ids, top_k, mode='hybrid', rerank=True, generate_answer=True):
    pipeline = get_pipeline()
    result = pipeline.ask(question, paper_ids=paper_ids, top_k=top_k,
                          mode=mode, rerank=rerank, generate_answer=generate_answer)
    formula_start = perf_counter()
    try:
        from rag.formula_store import FormulaStore
        store = FormulaStore(pipeline.root)
        if not store.records:
            return {**result, 'formulas': [], 'formula_retrieval': {'vector_index': False, 'reason': 'no_authoritative_artifacts'}}
        manifest = current_manifest(pipeline.settings, pipeline.root)
        scope = paper_ids or result.get('route', {}).get('paper_ids')
        if hasattr(pipeline,'attach_formulas'):
            result.update(pipeline.attach_formulas(question,result['evidence'],paper_ids=scope,
                documents=manifest['documents'] if manifest else None,store=store,
                facets=result.get('query_plan',{}).get('facets',[])))
        else:
            result.update(attach_formulas(question, result['evidence'], paper_ids=scope,
                                         documents=manifest['documents'] if manifest else None,
                                         root=pipeline.root, store=store))
    except (ValueError, OSError, KeyError) as error:
        result.update(formulas=[], formula_retrieval={'vector_index': False, 'reason': 'formula_artifact_validation_failed',
                                                     'error_type': type(error).__name__})
    formula_ms = round((perf_counter()-formula_start)*1000, 2)
    result.setdefault('timings', {})['formula_join_ms'] = formula_ms
    if 'elapsed_ms' in result:
        result['elapsed_ms'] += round(formula_ms)
    return result

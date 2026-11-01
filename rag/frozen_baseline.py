"""Persist the chosen retrieval specification before accessing heldout scores."""
import hashlib
from datetime import datetime, timezone

from rag.config import ROOT
from rag.ingestion import current_manifest, fingerprint
from rag.routing import PaperRouter
from rag.storage import atomic_json, read_json


def freeze_baseline(settings, root=ROOT):
    if (settings.corpus, settings.chunk_strategy, settings.candidate_k,
        settings.rerank_k, settings.rrf_k, settings.context_char_budget) != ('papers', 'token-window', 100, 20, 60, 8000):
        raise ValueError('冻结方案要求固定分块、每路100条召回、20条重排、RRF60和8000字符预算。')
    manifest = current_manifest(settings, root)
    if not manifest or manifest['profile'] != settings.index_profile():
        raise ValueError('当前索引与冻结配置不匹配。')
    paths = ['rag/config.py', 'rag/pipeline.py', 'rag/retrieval.py', 'rag/routing.py',
             'rag/embeddings.py', 'rag/generation.py', 'rag/benchmark.py', 'rag/frozen_baseline.py',
             'scripts/accept_heldout.py', 'configs/paper_aliases.json']
    code = {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}
    router = PaperRouter.from_project(manifest['documents'], root)
    spec = {'settings': settings.model_dump(), 'snapshot_id': manifest['snapshot_id'],
            'documents': len(manifest['documents']), 'chunks': manifest['chunks'],
            'documents_digest': fingerprint(manifest['documents']),
            'catalog_fingerprint': router.fingerprint, 'code_sha256': code,
            'lexical_policy_for_acceptance': 'fixed benchmark English translation',
            'live_lexical_policy': 'explicit lexical_query, otherwise original question',
            'routing': 'explicit user selection, otherwise conservative single-paper router',
            'rerank_input': 'title + body; original bodies retained',
            'context_policy': 'select_context over all 20 reranked candidates',
            'top_k': 5, 'query_decomposition': False, 'facet_reranking': False}
    digest = fingerprint(spec)
    path = root/'runtime/acceptance/frozen_baseline.json'
    previous = read_json(path)
    if previous:
        if previous['spec_sha256'] != digest or previous['spec'] != spec:
            raise ValueError('代码、目录或配置已偏离冻结方案。保留原验收结果，先核对变更。')
        return previous
    record = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'spec_sha256': digest, 'spec': spec}
    atomic_json(path, record)
    return record

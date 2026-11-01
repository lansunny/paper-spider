"""Load database/index settings without storing credentials in project files."""
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    qdrant_url: str = 'http://127.0.0.1:6333'
    collection: str = Field(default='paperrag_bge_m3_v1', pattern=r'^[a-zA-Z0-9_-]+$')
    vector_name: str = 'dense'
    vector_size: int = Field(default=1024, gt=0)
    embedding_model: str = 'BAAI/bge-m3'
    embedding_revision: str | None = None
    embedding_path: str = 'runtime/models/embedding'
    reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    reranker_revision: str = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    reranker_path: str = 'runtime/models/reranker'
    corpus: str = Field(default='papers', pattern=r'^(papers|qasper-train|qasper-dev)$')
    chunk_size: int = Field(default=384, ge=32)
    chunk_overlap: int = Field(default=64, ge=0)
    chunk_strategy: str = Field(default='token-window', pattern=r'^(token-window|layout-blocks)$')
    block_merge_gap: float = Field(default=8.0, ge=0, le=30)
    small_block_tokens: int = Field(default=80, ge=0, le=256)
    embedding_max_length: int = Field(default=512, ge=64)
    reranker_max_length: int = Field(default=768, ge=64)
    device: str = 'auto'
    batch_size: int = Field(default=8, ge=1)
    reranker_batch_size: int = Field(default=4, ge=1)
    candidate_k: int = Field(default=30, ge=1, le=200)
    rerank_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1)
    context_char_budget: int = Field(default=16000, ge=1000)
    schema_version: int = Field(default=1, gt=0)
    hnsw_m: int = Field(default=16, ge=4)
    hnsw_ef_construct: int = Field(default=100, ge=4)
    indexing_threshold_kb: int = Field(default=1000, ge=0)
    timeout_seconds: int = Field(default=15, gt=0)

    @model_validator(mode='after')
    def valid_chunk_budget(self):
        if self.chunk_strategy == 'layout-blocks' and (self.chunk_overlap != 0 or self.corpus != 'papers'):
            raise ValueError('layout-blocks requires papers corpus and zero token overlap')
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        if self.chunk_size + 2 > self.embedding_max_length:
            raise ValueError('chunk_size plus special tokens exceeds embedding_max_length')
        return self

    @property
    def chunker_version(self) -> str:
        if self.chunk_strategy == 'layout-blocks':
            return f'layout-blocks-{self.chunk_size}-gap{self.block_merge_gap:g}-small{self.small_block_tokens}-v1'
        return f'token-window-{self.chunk_size}-{self.chunk_overlap}-v1'

    def index_profile(self) -> dict:
        profile = {**self.model_dump(include={
            'vector_name', 'vector_size', 'embedding_model', 'embedding_revision',
            'schema_version', 'corpus', 'embedding_max_length',
        }), 'chunker_version': self.chunker_version, 'pooling': 'cls', 'normalized': True}
        if self.chunk_strategy == 'layout-blocks':
            profile.update(source_normalization='normalized-blocks-v1', layout_version='column-bands-v1',
                           provenance_version='source-spans-v1', embedding_input='body-v1')
        return profile


def load_settings(path: Path | None = None) -> RetrievalSettings:
    values = json.loads((path or ROOT / 'configs/retrieval.json').read_text(encoding='utf-8'))
    for key, variable in [('qdrant_url', 'QDRANT_URL'), ('collection', 'QDRANT_COLLECTION')]:
        if os.environ.get(variable):
            values[key] = os.environ[variable]
    return RetrievalSettings(**values)

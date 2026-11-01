"""Qdrant named dense vectors, version checks, snapshot isolation and filtering."""
import os
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from rag.config import RetrievalSettings
from rag.schema import Chunk


class QdrantStore:
    def __init__(self, settings: RetrievalSettings, client=None):
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url, api_key=os.getenv('QDRANT_API_KEY') or None,
            timeout=settings.timeout_seconds, prefer_grpc=False, trust_env=False,
        )

    def ensure_collection(self):
        cfg = self.settings
        if not cfg.embedding_revision:
            raise ValueError('Pin embedding_revision before creating a searchable index.')
        if not self.client.collection_exists(cfg.collection):
            self.client.create_collection(
                cfg.collection,
                vectors_config={cfg.vector_name: models.VectorParams(
                    size=cfg.vector_size, distance=models.Distance.COSINE)},
                hnsw_config=models.HnswConfigDiff(m=cfg.hnsw_m, ef_construct=cfg.hnsw_ef_construct),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=cfg.indexing_threshold_kb),
                metadata={'paperrag_profile': cfg.index_profile()},
            )
        self.validate_collection()
        existing = self.client.get_collection(cfg.collection).payload_schema
        for field in ['paper_id', 'snapshot_id', 'corpus', 'document_sha256']:
            if field not in existing:
                self.client.create_payload_index(cfg.collection, field,
                                                 models.PayloadSchemaType.KEYWORD, wait=True)

    def validate_collection(self):
        info = self.client.get_collection(self.settings.collection)
        vectors = info.config.params.vectors
        vector = vectors.get(self.settings.vector_name) if isinstance(vectors, dict) else None
        if vector is None or vector.size != self.settings.vector_size or vector.distance != models.Distance.COSINE:
            raise ValueError('Qdrant vector dimensions/name/distance mismatch; use a new collection.')
        if (info.config.metadata or {}).get('paperrag_profile') != self.settings.index_profile():
            raise ValueError('Index model/chunker profile mismatch; use a new collection for this experiment.')

    def _vectors(self, values, expected_rows=None):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.settings.vector_size:
            raise ValueError(f'Expected vectors shaped (n, {self.settings.vector_size}).')
        if expected_rows is not None and len(array) != expected_rows:
            raise ValueError('Chunk/vector count mismatch.')
        if not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
            raise ValueError('Vectors must be finite and nonzero.')
        return array

    @staticmethod
    def snapshot_filter(snapshot_id, paper_ids=None, corpus=None):
        conditions = [models.FieldCondition(key='snapshot_id', match=models.MatchValue(value=snapshot_id))]
        if paper_ids:
            conditions.append(models.FieldCondition(key='paper_id', match=models.MatchAny(any=list(paper_ids))))
        if corpus:
            conditions.append(models.FieldCondition(key='corpus', match=models.MatchValue(value=corpus)))
        return models.Filter(must=conditions)

    def upsert(self, chunks: list[Chunk], vectors, snapshot_id: str):
        array = self._vectors(vectors, len(chunks))
        if not chunks:
            return
        points = []
        for chunk, vector in zip(chunks, array, strict=True):
            if chunk.chunker_version != self.settings.chunker_version or chunk.corpus != self.settings.corpus:
                raise ValueError('Chunk corpus/version does not match the index profile.')
            points.append(models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, snapshot_id + ':' + chunk.chunk_id)),
                vector={self.settings.vector_name: vector.tolist()},
                payload={**chunk.payload(), 'snapshot_id': snapshot_id},
            ))
        self.client.upsert(self.settings.collection, points, wait=True)

    def count(self, snapshot_id):
        return self.client.count(self.settings.collection,
                                 count_filter=self.snapshot_filter(snapshot_id), exact=True).count

    def search(self, vector, snapshot_id, top_k=30, paper_ids=None, exact=False):
        if top_k < 1:
            raise ValueError('top_k must be positive.')
        query = self._vectors([vector])[0]
        result = self.client.query_points(
            self.settings.collection, query=query.tolist(), using=self.settings.vector_name,
            query_filter=self.snapshot_filter(snapshot_id, paper_ids, self.settings.corpus),
            search_params=models.SearchParams(hnsw_ef=128, exact=exact),
            limit=top_k, with_payload=True, with_vectors=False,
        )
        return [{**point.payload, 'score': float(point.score),
                 'scores': {'dense': float(point.score)}} for point in result.points]

    def close(self):
        self.client.close()

"""Lazy local BGE models. A lock bounds GPU concurrency on a 6 GB laptop."""
import json
from pathlib import Path
from threading import RLock

import numpy as np

from rag.config import ROOT, RetrievalSettings


class LocalModels:
    def __init__(self, settings: RetrievalSettings, root: Path = ROOT):
        self.settings, self.root = settings, root
        self.lock = RLock()
        self._encoder = self._reranker = None

    def _path(self, kind):
        cfg = self.settings
        path = self.root / (cfg.embedding_path if kind == 'embedding' else cfg.reranker_path)
        marker = path / 'paperrag-model.json'
        if not marker.exists():
            raise RuntimeError('本地模型尚未准备好，请运行 main.py --task models。')
        spec = json.loads(marker.read_text(encoding='utf-8'))
        model_id = cfg.embedding_model if kind == 'embedding' else cfg.reranker_model
        revision = cfg.embedding_revision if kind == 'embedding' else cfg.reranker_revision
        if spec.get('model_id') != model_id or spec.get('revision') != revision:
            raise ValueError(f'{kind} model revision mismatch; download the configured revision first.')
        return path

    def _device(self):
        import torch
        return ('cuda' if torch.cuda.is_available() else 'cpu') if self.settings.device == 'auto' else self.settings.device

    @property
    def encoder(self):
        with self.lock:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(str(self._path('embedding')),
                                                    device=self._device(), trust_remote_code=False)
                self._encoder.max_seq_length = self.settings.embedding_max_length
                if self._device().startswith('cuda'):
                    self._encoder.half()
            return self._encoder

    @property
    def tokenizer(self):
        return self.encoder.tokenizer

    def encode(self, texts):
        with self.lock:
            lengths = self.encoder.tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']
            if any(len(ids) > self.settings.embedding_max_length for ids in lengths):
                raise ValueError('Embedding input exceeds encoding limit; silent truncation is disabled.')
            values = self.encoder.encode(texts, batch_size=self.settings.batch_size,
                                         normalize_embeddings=True, convert_to_numpy=True,
                                         show_progress_bar=False)
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (len(texts), self.settings.vector_size):
                raise ValueError('Embedding model returned unexpected vector dimensions.')
            if not np.isfinite(values).all():
                raise ValueError('Embedding produced non-finite values.')
            return values

    def query_vector(self, question):
        if len(self.tokenizer.encode(question)) > self.settings.embedding_max_length:
            raise ValueError('问题超出当前向量模型的编码长度，请缩短问题。')
        return self.encode([question])[0]

    def rerank(self, question, hits):
        if not hits:
            return []
        with self.lock:
            import torch
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                self._reranker = CrossEncoder(str(self._path('reranker')), device=self._device(),
                                              max_length=self.settings.reranker_max_length,
                                              trust_remote_code=False)
                if self._device().startswith('cuda'):
                    self._reranker.model.half()
            values = self._reranker.predict(
                [(question, hit['text']) for hit in hits],
                batch_size=self.settings.reranker_batch_size,
                activation_fct=torch.nn.Identity(), show_progress_bar=False,
            )
        ranked = [{**hit, 'score': float(score), 'scores': {**hit.get('scores', {}), 'rerank': float(score)}}
                  for hit, score in zip(hits, values, strict=True)]
        if any(not np.isfinite(hit['score']) for hit in ranked):
            raise ValueError('Reranker produced non-finite values.')
        return sorted(ranked, key=lambda hit: (-hit['score'], hit['chunk_id']))

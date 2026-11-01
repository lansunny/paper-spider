"""Regression tests for index consistency, ranking, provenance and the API contract."""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import numpy as np
import pymupdf

from rag.chunking import chunk_text
from rag.config import RetrievalSettings
from rag.generation import GenerationSettings, generate, select_context, validate_answer
from rag.ingestion import build_index, current_chunks, current_manifest
from rag.retrieval import reciprocal_rank_fusion, retrieval_metrics
from rag.schema import Chunk
from rag.storage import atomic_json


def settings(**overrides):
    return RetrievalSettings(embedding_revision='fixture-revision', vector_size=3,
                             chunk_size=32, chunk_overlap=8, embedding_max_length=64, **overrides)


def chunk(paper='p1', text='Faithfulness supports grounded answers.'):
    return Chunk(paper_id=paper, document_sha256='a' * 64, chunker_version=settings().chunker_version,
                 pdf_page=1, chunk_index=0, text=text)


class Tokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(match.start(), match.end()) for match in re.finditer(r'\S+', text)]
        return {'input_ids': list(range(len(offsets))), 'offset_mapping': offsets}


class FakeModels:
    tokenizer = Tokenizer()

    def encode(self, texts):
        return np.asarray([[1., 0., 0.]] * len(texts), dtype=np.float32)


class MemoryStore:
    def __init__(self):
        self.snapshots = {}

    def ensure_collection(self):
        pass

    def count(self, snapshot):
        return len(self.snapshots.get(snapshot, {}))

    def upsert(self, chunks, vectors, snapshot):
        self.snapshots.setdefault(snapshot, {}).update({c.chunk_id: c for c in chunks})


class CoreTests(unittest.TestCase):
    def test_web_request_exposes_experiment_switches_and_source_page(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        paper = {'paper_id': 'p1', 'title': 'Fixture', 'indexed': True}
        result = {'answer': None, 'evidence': [{**chunk().payload(), 'score': 1.}]}
        with patch('backend.app.papers', return_value=[paper]), patch('backend.rag_service.ask', return_value=result) as ask:
            client = TestClient(app)
            response = client.post('/api/ask', json={'question': 'faithfulness', 'paper_ids': ['p1'],
                                                    'top_k': 3, 'mode': 'dense', 'rerank': False,
                                                    'generate_answer': False})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['evidence'][0]['pdf_url'], '/api/papers/p1/pdf#page=1')
            ask.assert_called_once_with('faithfulness', ['p1'], 3, 'dense', False, False)
            self.assertEqual(client.post('/api/ask', json={'question': 'q', 'mode': 'invalid'}).status_code, 422)

    def test_evaluation_entry_writes_a_limited_report_without_generation(self):
        from rag.evaluation import evaluate
        class FixturePipeline:
            def retrieve(self, *args):
                return {'evidence': [{'chunk_id': 'c1', 'paragraph_ids': ['paragraph1']}]}
        with tempfile.TemporaryDirectory(prefix='paperrag-eval-test-') as temporary:
            root = Path(temporary)
            p = FixturePipeline()
            p.settings, p.root = settings(corpus='qasper-train'), root
            atomic_json(root / 'runtime/indexes' / p.settings.collection / 'CURRENT.json', {'snapshot_id': 'fixture'})
            data = {'question_id': 'q1', 'paper_id': 'p1', 'question': 'fixture?', 'annotations': [
                {'text_retrieval_eligible': True, 'evidence_paragraph_groups': [['paragraph1']]}]}
            (root / 'data/evaluation').mkdir(parents=True)
            (root / 'data/evaluation/starter_train.jsonl').write_text(json.dumps(data) + '\n', encoding='utf-8')
            path, result = evaluate(p, ['bm25'], limit=1, progress=lambda s: None)
            self.assertEqual(result['bm25']['evidence_recall_at_5_best_annotation'], 1.)
            self.assertTrue(json.loads(path.read_text())['limited_run'])

    def test_chunk_identity_and_boundaries(self):
        self.assertEqual(chunk().chunk_id, chunk().chunk_id)
        self.assertNotEqual(chunk().chunk_id, chunk(text='changed').chunk_id)
        text = ' '.join(f'word{i}' for i in range(100))
        chunks = list(chunk_text(text, Tokenizer(), settings(), paper_id='p',
                                 document_sha256='a' * 64, pdf_page=2))
        self.assertTrue(all(len(c.text.split()) <= 32 for c in chunks))
        self.assertEqual(chunks[0].text.split()[-8:], chunks[1].text.split()[:8])
        self.assertEqual(set(' '.join(c.text for c in chunks).split()), set(text.split()))
        self.assertTrue(all(c.pdf_page == 2 for c in chunks))

    def test_rrf_uses_rank_and_deduplicates(self):
        a = {'chunk_id': 'a', 'text': 'A', 'score': 1000, 'scores': {'bm25': 1000}}
        b = {'chunk_id': 'b', 'text': 'B', 'score': .9, 'scores': {'dense': .9}}
        result = reciprocal_rank_fusion({'bm25': [a, a, b], 'dense': [b, a]})
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(next(x for x in result if x['chunk_id'] == 'a')['score'], 1/61 + 1/62)
        self.assertLess(result[0]['score'], 1)

    def test_context_budget_and_citation_checks(self):
        evidence, used = select_context([{**chunk().payload(), 'score': 1}], 1000)
        self.assertLessEqual(used, 1000)
        valid = json.dumps({'answer': '有依据。[S1]', 'citations': ['S1'], 'insufficient_evidence': False})
        self.assertEqual(validate_answer(valid, evidence).citations, ['S1'])
        for content in [valid.replace('S1', 'S9'),
                        json.dumps({'answer': '没有引用', 'citations': [], 'insufficient_evidence': False})]:
            with self.assertRaises(ValueError):
                validate_answer(content, evidence)

    def test_deepseek_request_and_response_without_real_credentials(self):
        evidence, _ = select_context([chunk().payload()], 1000)
        def handle(request):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
            self.assertEqual(body['model'], 'deepseek-v4-pro')
            self.assertEqual(body['thinking']['type'], 'disabled')
            self.assertIn('evidence', body['messages'][1]['content'])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'answer': '有依据。[S1]', 'citations': ['S1'], 'insufficient_evidence': False})}}],
                'usage': {'prompt_tokens': 30, 'completion_tokens': 12}})
        result = generate('问题', evidence, GenerationSettings(api_key='test-only'), httpx.MockTransport(handle))
        self.assertEqual(result['usage']['completion_tokens'], 12)
        self.assertEqual(result['citation_validation'], 'source_ids_and_inline_consistency_only')

    def test_truncated_generation_is_not_accepted(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}))
        with self.assertRaises(ValueError):
            generate('question', [], GenerationSettings(api_key='test-only'), transport)

    def test_evidence_metric_does_not_double_count_overlapping_chunks(self):
        annotations = [{'text_retrieval_eligible': True, 'evidence_paragraph_groups': [['p1'], ['p2']]}]
        hits = [{'paragraph_ids': ['p1']}, {'paragraph_ids': ['p1']}]
        self.assertEqual(retrieval_metrics(hits, annotations), {'evidence_recall': .5, 'mrr': 1.})
        self.assertIsNone(retrieval_metrics(hits, []))

    def test_rebuild_failure_preserves_old_snapshot_and_reuses_unchanged_index(self):
        with tempfile.TemporaryDirectory(prefix='paperrag-unit-') as temporary:
            root = Path(temporary)
            (root / 'data/pdfs').mkdir(parents=True)
            pdf_path = root / 'data/pdfs/test.pdf'
            def write_pdf(word):
                with pymupdf.open() as pdf:
                    page = pdf.new_page()
                    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), ' '.join([word] * 120), fontsize=10)
                    pdf.save(pdf_path)
            write_pdf('alpha')
            atomic_json(root / 'data/paper_manifest.json', [{'paper_id': 'p1', 'status': 'downloaded',
                        'pdf_path': 'data/pdfs/test.pdf', 'title': 'Fixture'}])
            store, model = MemoryStore(), FakeModels()
            original = build_index(settings(), model, store, root)
            self.assertTrue(build_index(settings(), model, store, root)['reused'])
            self.assertGreater(len(current_chunks(settings(), root)[1]), 0)
            write_pdf('beta')
            with patch.object(model, 'encode', side_effect=RuntimeError('simulated inference failure')):
                with self.assertRaises(RuntimeError):
                    build_index(settings(), model, store, root)
            self.assertEqual(current_manifest(settings(), root)['snapshot_id'], original['snapshot_id'])


@unittest.skipUnless(os.getenv('PAPERRAG_TEST_QDRANT'), 'Set PAPERRAG_TEST_QDRANT to run real database checks.')
class QdrantIntegrationTests(unittest.TestCase):
    def test_persistence_filters_profiles_and_idempotence(self):
        from rag.vector_store import QdrantStore
        cfg = settings(collection='paperrag_test_' + uuid4().hex,
                       qdrant_url=os.environ['PAPERRAG_TEST_QDRANT'])
        store = QdrantStore(cfg)
        created = False
        try:
            self.assertFalse(store.client.collection_exists(cfg.collection))
            store.ensure_collection()
            created = True
            chunks = [chunk('p1'), chunk('p2', 'Dense retrieval')]
            vectors = [[1., 0., 0.], [0., 1., 0.]]
            store.upsert(chunks, vectors, 'old')
            store.upsert(chunks, vectors, 'old')
            self.assertEqual(store.count('old'), 2)
            self.assertEqual(store.search([1., 0., 0.], 'old', exact=True)[0]['paper_id'], 'p1')
            selected = store.search([1., 0., 0.], 'old', paper_ids=['p2'], exact=True)
            self.assertEqual([h['paper_id'] for h in selected], ['p2'])
            store.upsert([chunks[0]], [[0., 0., 1.]], 'new')
            self.assertEqual(store.count('old'), 2)
            self.assertEqual(store.count('new'), 1)
            with self.assertRaises(ValueError):
                store.upsert(chunks, [[0., 0., 0.], [0., 1., 0.]], 'invalid')
            with self.assertRaises(ValueError):
                store.search([1., 0.], 'old')
            wrong = QdrantStore(cfg.model_copy(update={'embedding_revision': 'different'}), client=store.client)
            with self.assertRaises(ValueError):
                wrong.validate_collection()
            store.close()
            store = QdrantStore(cfg)
            self.assertEqual(store.count('old'), 2)
        finally:
            if created and cfg.collection.startswith('paperrag_test_'):
                store.client.delete_collection(cfg.collection)
            store.close()


if __name__ == '__main__':
    unittest.main()

"""Graph anchors, API boundaries, caching and sparse-community behavior; no paid API."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pymupdf

from backend.paper_graph import create_paper_graph_router
from rag.config import RetrievalSettings
from rag.generation import GenerationSettings
from rag.ingestion import fingerprint
from rag.paper_graph.build import GraphSettings, build_bundle, prepare, profile_cache
from rag.paper_graph.extraction import request_json
from rag.paper_graph.graph import similarity_graph
from rag.paper_graph.schema import DIMENSIONS, FIELDS, SourceAnchor, validate_description
from rag.paper_graph.sources import load_sources, source_batches
from rag.paper_graph.store import PaperGraphStore
from rag.storage import atomic_json


def description(source_id, text='A method with a specific retrieval trigger.'):
    return {dimension: {'status': 'available', 'description': text,
        'claims': [{'field': FIELDS[dimension][0], 'text': text, 'source_ids': [source_id]}], 'reason': ''}
        for dimension in DIMENSIONS}


class FakeModels:
    def encode(self, texts):
        result = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            result[index, :2] = [1, .1 if 'trigger' in text else .2]
        return result / np.linalg.norm(result, axis=1, keepdims=True)


class PaperGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.documents = []
        for index in range(3):
            path = self.root / f'p{index}.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((70, 90), f'Paper {index}. Retrieval trigger and evaluation settings.')
                page.insert_text((70, 140), 'The answer is checked against supporting evidence.')
                pdf.save(path)
            self.documents.append({'paper_id': f'p{index}', 'title': f'Paper {index}',
                'pdf_path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'publication_year': 2024, 'status': 'downloaded'})
        atomic_json(self.root / 'data/paper_manifest.json', self.documents)
        self.config = GraphSettings(limit=3, thresholds={d: .5 for d in DIMENSIONS})
        self.generation = GenerationSettings(api_key='test-only')

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        records = prepare(self.root, self.documents, self.config, self.generation)
        for record in records:
            dimensions = description(record['sources'][0]['source_id'])
            atomic_json(profile_cache(self.root, record), {'cache_key': record['cache_key'],
                'document_sha256': record['paper']['sha256'], 'model': 'test-fixture',
                'dimensions': dimensions, 'profile_sha256': fingerprint(dimensions), 'requests': []})
        return build_bundle(self.root, records, self.config, RetrievalSettings(), FakeModels(), 'fixture-snapshot')

    def test_source_anchors_are_exact_and_independent_of_chunks(self):
        document = self.documents[0]
        sources, _ = load_sources(document, self.root)
        again, _ = load_sources({**document, 'chunker_version': 'a-different-chunker'}, self.root)
        self.assertEqual(sources, again)
        cached = self.root / 'runtime/parsed' / (document['sha256'] + '.jsonl')
        from rag.chunking import normalized_blocks
        pages = [json.loads(line) for line in cached.read_text(encoding='utf-8').splitlines()]
        for source in sources:
            canonical, _ = normalized_blocks(pages[source['pdf_page'] - 1]['blocks'])
            self.assertEqual(canonical[source['source_start']:source['source_end']], source['text'])
            SourceAnchor.model_validate(source)

    def test_changed_pdf_is_rejected_before_extraction(self):
        (self.root / self.documents[0]['pdf_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            load_sources(self.documents[0], self.root)

    def test_long_blocks_split_without_losing_text_or_offsets(self):
        sources, _ = load_sources(self.documents[0], self.root)
        source = sources[0]
        text = 'A数学B ' * 2000
        source.update(text=text, source_end=source['source_start'] + len(text),
                      text_sha256=hashlib.sha256(text.encode()).hexdigest())
        cards, batches = source_batches([source], 4000)
        self.assertGreater(len(batches), 1)
        self.assertEqual(''.join(c['text'] for c in cards), text)
        for card in cards:
            SourceAnchor.model_validate(card)
            self.assertEqual(text[card['source_start'] - source['source_start']:
                card['source_end'] - source['source_start']], card['text'])

    def test_unknown_sources_fields_and_inconsistent_missing_status_are_rejected(self):
        value = description('S1')
        validate_description(value, {'S1'})
        with self.assertRaises(ValueError):
            validate_description(value, {'S2'})
        invalid = copy.deepcopy(value)
        invalid['method']['claims'][0]['field'] = 'invented_field'
        with self.assertRaises(ValueError):
            validate_description(invalid, {'S1'})
        invalid = copy.deepcopy(value)
        invalid['method']['status'] = 'not_found_in_input'
        with self.assertRaises(ValueError):
            validate_description(invalid, {'S1'})

    def test_model_step_cache_reuses_validated_response_without_another_call(self):
        calls = []
        value = description('S1')
        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={'model': 'fixture', 'usage': {'total_tokens': 25},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})
        validator = lambda v: validate_description(v, {'S1'})
        kwargs = {'transport': httpx.MockTransport(respond)}
        first, record = request_json('test', {'sources': []}, self.generation, self.root / 'cache', validator, **kwargs)
        second, cached = request_json('test', {'sources': []}, self.generation, self.root / 'cache', validator, **kwargs)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertEqual(record, cached)

    def test_short_model_ids_resolve_to_immutable_source_ids(self):
        canonical = 'S0123456789abcdefabcd'
        def respond(request):
            payload = json.loads(json.loads(request.content)['messages'][1]['content'])
            self.assertEqual(payload['sources'][0]['id'], 'S1')
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(description('S1'))}}]})
        value, record = request_json('test', {'sources': [{'id': canonical, 'page': 1, 'text': 'Evidence'}]},
            self.generation, self.root / 'cache', lambda v: validate_description(v, {canonical}),
            transport=httpx.MockTransport(respond))
        self.assertEqual(value.method.claims[0].source_ids, [canonical])
        self.assertEqual(record['value']['method']['claims'][0]['source_ids'], [canonical])
        self.assertEqual(record['source_aliases'], {'S1': canonical})

    def test_knn_keeps_isolated_papers_and_deterministic_communities(self):
        vectors = np.array([[1, 0, 0], [.99, .01, 0], [0, 1, 0], [.01, .99, 0], [0, 0, 1]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        kwargs = dict(dimension='topic', neighbors=1, threshold=.9, resolution=1, seed=42)
        graph = similarity_graph(list('abcde'), vectors, **kwargs)
        self.assertEqual(graph, similarity_graph(list('abcde'), vectors, **kwargs))
        self.assertEqual(graph['statistics']['edges'], 2)
        self.assertEqual(graph['statistics']['communities'], 2)
        self.assertEqual(graph['nodes'][-1]['degree'], 0)
        self.assertIsNone(graph['nodes'][-1]['community_id'])
        self.assertTrue(all(e['source'] != e['target'] for e in graph['edges']))

    def test_failed_step_resumes_with_repair_and_keeps_usage_of_rejected_outputs(self):
        invalid = description('S1')
        invalid['method']['claims'][0]['source_ids'] = []
        def response(value):
            return httpx.Response(200, json={'usage': {'total_tokens': 1},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]})
        validator = lambda value: validate_description(value, {'S1'})
        with self.assertRaises(ValueError):
            request_json('test', {}, self.generation, self.root / 'cache', validator,
                transport=httpx.MockTransport(lambda request: response(invalid)))
        def repaired(request):
            messages = json.loads(request.content)['messages']
            self.assertIn('省略对应 claim', messages[-1]['content'])
            return response(description('S1'))
        value, record = request_json('test', {}, self.generation, self.root / 'cache', validator,
            transport=httpx.MockTransport(repaired))
        self.assertEqual(value.method.claims[0].source_ids, ['S1'])
        self.assertEqual(len(record['attempts']), 4)

    def test_empty_dimension_does_not_create_placeholder_vectors_or_edges(self):
        graph = similarity_graph([], np.empty((0, 1024)), dimension='experiment',
            neighbors=10, threshold=.5, resolution=1, seed=42)
        self.assertEqual(graph['nodes'], [])
        self.assertEqual(graph['edges'], [])

    def test_published_graph_api_returns_traceable_sources_and_neighbors(self):
        directory, manifest = self.build()
        app = FastAPI()
        app.include_router(create_paper_graph_router(self.root))
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/paper-graph').json()['papers'], 3)
            paper = client.get('/api/paper-graph/papers/p0').json()
            source = paper['dimensions']['method']['claims'][0]['sources'][0]
            self.assertEqual(source['document_sha256'], self.documents[0]['sha256'])
            self.assertIn('/study?paper=p0&page=1', source['study_url'])
            self.assertEqual(client.get('/api/paper-graph/sources/' + source['source_id']).json(), source)
            view = client.get('/api/paper-graph/views/method').json()
            self.assertEqual(len(view['nodes']), 3)
            neighbors = client.get('/api/paper-graph/papers/p0/neighbors?dimension=method').json()['neighbors']
            self.assertEqual(len(neighbors), 2)
            self.assertEqual(client.get('/api/paper-graph/views/invalid').status_code, 422)
            self.assertEqual(client.get('/api/paper-graph/papers/missing').status_code, 404)
            self.assertEqual(manifest['validation'], 'pipeline_smoke_run_not_quality_benchmark')
            self.assertEqual(manifest['retrieval_integration'], 'not_enabled')
            self.assertEqual(source['source_validation'], 'matched_current_pdf_and_canonical_text')
            (self.root / self.documents[0]['pdf_path']).write_bytes(b'changed PDF version')
            self.assertEqual(client.get('/api/paper-graph/sources/' + source['source_id']).status_code, 409)
            with (directory / 'graph.sqlite3').open('ab') as stream:
                stream.write(b'tampered')
            self.assertEqual(client.get('/api/paper-graph/views/topic').status_code, 409)

    def test_missing_profile_cannot_replace_a_published_graph(self):
        self.build()
        before = PaperGraphStore(self.root).summary()['build_id']
        records = prepare(self.root, self.documents, self.config, self.generation)
        profile_cache(self.root, records[0]).unlink()
        with self.assertRaises(ValueError):
            build_bundle(self.root, records, self.config, RetrievalSettings(), FakeModels(), 'fixture-snapshot')
        self.assertEqual(PaperGraphStore(self.root).summary()['build_id'], before)


if __name__ == '__main__':
    unittest.main()

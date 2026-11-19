import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag.docling_search import DoclingSearch, direct_object_query, fuse_objects, nearest_figure
from rag.query_translation import Translation


class DoclingSearchTest(unittest.TestCase):
    def test_nearby_respects_page_column_and_document_version(self):
        def item(x, y, *, page=4, digest='same'):
            return {'paper_id': 'paper', 'pdf_sha256': digest, 'headings': ['Methods'], 'caption': 'Retrieval architecture',
                    'provenance': [{'page_no': page, 'bbox': {'l': x, 'r': x+160, 't': y, 'b': y+55,
                                                          'coord_origin': 'TOPLEFT'}}]}

        paragraph = item(80, 120)
        self.assertEqual(nearest_figure(paragraph, item(90, 185), {'retrieval'})[1], 4)
        self.assertIsNone(nearest_figure(paragraph, item(320, 185), {'retrieval'}))
        self.assertIsNone(nearest_figure(paragraph, item(90, 185, page=5), {'retrieval'}))
        self.assertIsNone(nearest_figure(paragraph, item(90, 185, digest='new'), {'retrieval'}))

    def test_repeated_rows_do_not_take_multiple_votes(self):
        rankings = {'row': [{'object_id': 'A', 'score': 100}, {'object_id': 'A'}, {'object_id': 'B'}],
                    'summary': [{'object_id': 'B', 'score': .01}]}
        results = fuse_objects(rankings, 60)
        self.assertEqual([row['object_id'] for row in results], ['B','A'])
        self.assertEqual(results[0]['paths']['row']['rank'], 2)
        self.assertAlmostEqual(results[1]['rrf_score'], 1/61)

    def test_direct_number_does_not_swallow_analysis_request(self):
        self.assertTrue(direct_object_query('Figure 2?'))
        self.assertTrue(direct_object_query('Table A.1'))
        self.assertTrue(direct_object_query('表 1？'))
        self.assertFalse(direct_object_query('Table 1 Which row reports EM?'))

    def test_query_language_routes_original_and_english_to_the_expected_rankers(self):
        query = 'ARES 需要多少标注？'
        english = 'How many annotations does ARES require?'
        engine = DoclingSearch.__new__(DoclingSearch)
        engine.profile = 'profile'
        engine.cfg = SimpleNamespace(rrf_k=60, rerank_k=20, context_char_budget=16000)
        engine.body_rerank_k = 100
        engine.manifest = {'snapshot_id': 'snapshot'}
        engine.lock = nullcontext()
        engine.connection = Mock()
        engine.text_model = Mock()
        engine.clip = Mock()
        engine.visual = Mock()
        engine.visual.return_value.rank.return_value = []
        engine._hybrid = Mock(return_value=[])
        with patch('rag.docling_search.QdrantStore'), patch('rag.docling_search.translate_query') as translate:
            translate.return_value = Translation(english, 'test-model', 10)
            engine.search(query)
            self.assertEqual([call.args[:2] for call in engine._hybrid.call_args_list],
                             [(branch, english) for branch in ('body', 'caption', 'tables', 'rows')])
            self.assertEqual([call.args[5] for call in engine._hybrid.call_args_list], [100, 20, 20, 20])
            engine.clip.text.assert_called_with(english)
            engine.search(query, kinds=('figure',), clip_text='Architecture diagram')
            engine.clip.text.assert_called_with('Architecture diagram')
            engine.search(query, kinds=('body',), query_language='bm25_english')
            engine.text_model.query_vector.assert_called_with(query)
            self.assertEqual(engine._hybrid.call_args.args[1], query)
            self.assertEqual(engine._hybrid.call_args.kwargs['lexical_text'], english)
            engine.search(query, kinds=('body',), query_language='english')
            engine.text_model.query_vector.assert_called_with(english)
            self.assertEqual(engine._hybrid.call_args.args[1], english)
            self.assertEqual(engine._hybrid.call_args.kwargs['lexical_text'], english)
            translate.side_effect = RuntimeError('unavailable')
            result = engine.search(query, kinds=('body',), query_language='english')
            engine.text_model.query_vector.assert_called_with(query)
            self.assertEqual(engine._hybrid.call_args.kwargs['lexical_text'], query)
            self.assertEqual(result['query_translation']['status'], 'fallback')
            translate.reset_mock()
            engine.search(english, kinds=('body',), query_language='english')
            engine.search('图 1', kinds=('figure',), visual=False, query_language='english')
            translate.assert_not_called()

    def test_body_preview_and_context_have_separate_limits(self):
        engine = DoclingSearch.__new__(DoclingSearch)
        engine.profile = 'profile'
        engine.cfg = SimpleNamespace(context_char_budget=16000)
        engine.body_rerank_k = 100
        engine.manifest = {'snapshot_id': 'snapshot'}
        engine.lock = nullcontext()
        engine.connection = Mock()
        engine.text_model = Mock()
        hits = [{'chunk_id': str(i), 'paper_id': 'paper', 'title': 'Paper', 'text': f"{i} " + "word "*70,
                 'pdf_page': 2} for i in range(100)]
        engine._hybrid = Mock(return_value=hits)
        engine._body_results = Mock(side_effect=lambda db, selected: selected)
        with patch('rag.docling_search.QdrantStore'):
            result = engine.search('English question', kinds=('body',), top_k=5)
        self.assertEqual(len(result['body']), 5)
        self.assertGreater(len(result['body_context']), 20)
        self.assertLess(len(result['body_context']), 100)
        self.assertLessEqual(result['body_context_chars'], 16000)
        self.assertEqual(result['body_context'][0]['citation_id'], 'S1')

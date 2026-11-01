import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from pydantic import SecretStr

from rag.config import RetrievalSettings
from rag.generation import GenerationSettings
from rag.query_decomposition import QueryPlan, Subquery, allocate_slots, load_or_create_plan
from rag.routing import PaperRouter
from scripts import experiment_decomposition as experiment


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.router = PaperRouter([
            {'paper_id': 'r', 'title': 'RAGAs: Automated Evaluation'},
            {'paper_id': 'a', 'title': 'ARES: Automated Evaluation'}], ['r', 'a'])
        self.settings = GenerationSettings(api_key=SecretStr('fixture-key'))
        self.question, self.english = 'RAGAs 如何计算指标？说明步骤和定义。', 'How does RAGAs calculate the metric? Explain steps and definition.'
        self.plan = QueryPlan(subqueries=[Subquery(zh='RAGAs 的计算步骤是什么？', en='What are the calculation steps in RAGAs?')], reason='Separate the steps.')

    def response(self, plan):
        return httpx.Response(200, json={'model': 'fixture-model', 'usage': {'total_tokens': 12},
            'choices': [{'finish_reason': 'stop', 'message': {'content': plan.model_dump_json()}}]})

    def test_cached_planning_uses_only_questions_and_excludes_credentials(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return self.response(self.plan)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            transport = httpx.MockTransport(handler)
            plan, first = load_or_create_plan(self.question, self.english, self.settings, self.router,
                                              create=True, root=root, transport=transport)
            cached, second = load_or_create_plan(self.question, self.english, self.settings, self.router,
                                                 create=True, root=root, transport=transport)
            self.assertEqual(plan, cached)
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            self.assertEqual(len(requests), 1)
            self.assertEqual(set(json.loads(requests[0]['messages'][1]['content'])), {'question', 'question_en'})
            self.assertNotIn('fixture-key', Path(first['cache_path']).read_text(encoding='utf-8'))

    def test_unprepared_preview_does_not_call_api(self):
        with tempfile.TemporaryDirectory() as name:
            handler = Mock(side_effect=AssertionError('Unexpected request'))
            plan, record = load_or_create_plan(self.question, self.english, self.settings, self.router,
                                               root=Path(name), transport=httpx.MockTransport(handler))
            self.assertIsNone(plan)
            self.assertFalse(record['cache_hit'])
            handler.assert_not_called()

    def test_new_paper_in_subquery_rejected_without_caching(self):
        bad = QueryPlan(subqueries=[Subquery(zh='ARES 如何计算？', en='How does ARES calculate?')], reason='bad')
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(RuntimeError, '校验'):
                load_or_create_plan(self.question, self.english, self.settings, self.router,
                                    create=True, root=Path(name), transport=httpx.MockTransport(lambda r: self.response(bad)))
            self.assertFalse(list(Path(name).rglob('*.json')))

    def test_http_failure_is_not_retried(self):
        handler = Mock(return_value=httpx.Response(429))
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 429'):
                load_or_create_plan(self.question, self.english, self.settings, self.router,
                                    create=True, root=Path(name), transport=httpx.MockTransport(handler))
        self.assertEqual(handler.call_count, 1)

    def test_budget_and_scope_are_preserved_for_each_facet(self):
        self.assertEqual(allocate_slots(100, 4), [50, 17, 17, 16])
        lexical, store, models = Mock(), Mock(), Mock()
        original = {name: [{'paper_id': 'r', 'chunk_id': f'original-{i}', 'text': str(i), 'scores': {}}
                           for i in range(100)] for name in ['bm25_en', 'dense']}
        def hits(top_k):
            return [{'paper_id': 'r', 'chunk_id': f'facet-{i}', 'text': str(i), 'scores': {}} for i in range(top_k)]
        lexical.search.side_effect = lambda query, *, top_k, paper_ids: hits(top_k)
        store.search.side_effect = lambda vector, snapshot, *, top_k, paper_ids, exact: hits(top_k)
        queries = [Subquery(zh=f'RAGAs 子问题{i}', en=f'RAGAs facet {i}') for i in range(3)]
        plan = QueryPlan(subqueries=queries, reason='three facets')
        branches, trace = experiment.retrieve_multiquery(lexical, store, models, 'snapshot',
            {'question': self.question, 'question_en': self.english}, self.router.route(self.question), plan, original)
        self.assertEqual(trace['returned_slots_per_route'], {'bm25_en': 100, 'dense': 100})
        self.assertTrue(all(len(hits) <= 100 for hits in branches.values()))
        self.assertEqual(models.query_vector.call_count, 3)
        for call in lexical.search.call_args_list + store.search.call_args_list:
            self.assertEqual(call.kwargs['paper_ids'], ['r'])
        # A global parent stays global even when a facet explicitly names one paper.
        lexical.reset_mock()
        experiment.retrieve_multiquery(lexical, store, models, 'snapshot',
            {'question': self.question, 'question_en': self.english}, {'mode': 'global', 'paper_ids': []}, plan, original)
        self.assertTrue(all(call.kwargs['paper_ids'] is None for call in lexical.search.call_args_list))

    def test_no_facets_reuses_original_rankings_without_model_calls(self):
        models = Mock()
        original = {'bm25_en': [], 'dense': []}
        branches, trace = experiment.retrieve_multiquery(Mock(), Mock(), models, 'snapshot',
            {'question': self.question, 'question_en': self.english}, self.router.route(self.question),
            QueryPlan(subqueries=[], reason='simple'), original)
        self.assertIs(branches, original)
        self.assertEqual(trace['slots_per_route'], [100])
        models.query_vector.assert_not_called()

    def test_mocked_full_experiment_saves_pair_and_exact_source_scores(self):
        cfg = RetrievalSettings()
        question = {'question_id': 'q', 'question': self.question, 'question_en': self.english,
                    'answerable': True, 'source_scope': 'named_papers', 'category': 'method',
                    'gold_paper_ids': ['r'], 'gold_sources_exhaustive': True,
                    'evidence_units': [{'unit_id': 'u', 'alternatives': [[
                        {'paper_id': 'r', 'document_sha256': 'sha', 'pdf_page': 1,
                         'char_start': 0, 'char_end': 15, 'quote': 'TARGET EVIDENCE'}]]}]}
        texts = ['TARGET EVIDENCE', *[f'Distractor {i}.' for i in range(5)]]
        hits = [{'paper_id': 'r', 'document_sha256': 'sha', 'pdf_page': 1, 'chunk_id': f'h{i}',
                 'title': 'RAGAs', 'text': text, 'scores': {}} for i, text in enumerate(texts)]
        original = [*hits[1:], hits[0]]
        lexical, store, models = Mock(), Mock(), Mock()
        lexical.search.side_effect = lambda query, **kwargs: original if query == self.english else hits
        models.query_vector.side_effect = lambda query: [0] if query == self.question else [1]
        models.rerank.side_effect = lambda query, candidates: candidates
        store.search.side_effect = lambda vector, snapshot, **kwargs: original if vector == [0] else hits
        store.count.return_value = 6
        manifest = {'snapshot_id': 'snapshot', 'documents': {'r': 'sha'}, 'profile': cfg.index_profile()}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            benchmark = root/'data/evaluation/paper_benchmark/dev.jsonl'
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text(json.dumps(question), encoding='utf-8')
            with patch.object(experiment, 'ROOT', root), \
                 patch.object(experiment, 'load_settings', return_value=cfg), \
                 patch.object(experiment, 'current_manifest', return_value=manifest), \
                 patch.object(experiment, 'current_chunks', return_value=(manifest, hits)), \
                 patch.object(experiment, 'load_questions', return_value=[question]), \
                 patch.object(experiment.PaperRouter, 'from_project', return_value=self.router), \
                 patch.object(experiment, 'generation_settings', return_value=self.settings), \
                 patch.object(experiment, 'load_or_create_plan', return_value=(self.plan, {'cache_hit': True})) as planner, \
                 patch.object(experiment, 'QdrantStore', return_value=store), \
                 patch.object(experiment, 'LexicalIndex', return_value=lexical), \
                 patch.object(experiment, 'LocalModels', return_value=models), \
                 patch.object(experiment, 'load_source_pages', return_value={('r', 'sha', 1): ' '.join(texts)}), \
                 patch('sys.argv', ['experiment_decomposition', '--run']), redirect_stdout(io.StringIO()):
                experiment.main()
            self.assertFalse(planner.call_args.kwargs['create'])
            self.assertEqual(planner.call_args.args[:2], (self.question, self.english))
            directory, = (root/'runtime/experiments').iterdir()
            report = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(report['paired_questions'][0]['delta_decomposed_minus_single']['after_top5'], 1)
            self.assertEqual(json.loads((directory/'run.json').read_text(encoding='utf-8'))['status'], 'completed')
            row = json.loads((directory/'q_decomposed.json').read_text(encoding='utf-8'))
            self.assertLessEqual(row['context_chars'], 8000)
            self.assertTrue(row['metrics']['after_top5']['all_required_evidence'])
            self.assertTrue(any(h['text'] == 'TARGET EVIDENCE' for h in row['reranked_candidates']))
            store.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

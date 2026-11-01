import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from rag.config import RetrievalSettings
from rag.frozen_baseline import freeze_baseline
from rag.generation import GenerationSettings
from rag.pipeline import Pipeline
from rag.routing import PaperRouter
from scripts import accept_heldout, learn_generation


class FrozenPipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RetrievalSettings(candidate_k=100, context_char_budget=8000)
        self.models, self.store, self.lexical = Mock(), Mock(), Mock()
        self.hits = [{'chunk_id': f'h{i}', 'paper_id': 'p1', 'document_sha256': 'sha',
                      'pdf_page': 1, 'title': 'RAGAs', 'text': f'body {i}', 'scores': {}}
                     for i in range(6)]
        self.lexical.search.return_value = self.store.search.return_value = self.hits
        self.models.query_vector.return_value = [1]
        self.models.rerank.side_effect = lambda query, hits: hits
        self.pipeline = Pipeline(self.cfg, models=self.models, store=self.store)
        self.pipeline._router = PaperRouter([{'paper_id': 'p1', 'title': 'RAGAs: Evaluation'}], ['p1'])
        self.manifest = {'snapshot_id': 'snapshot', 'documents': {'p1': 'sha', 'p2': 'other'},
                         'chunks': 6, 'profile': self.cfg.index_profile()}

    def test_real_pipeline_uses_route_english_bm25_and_title_rerank(self):
        with patch.object(self.pipeline, 'snapshot', return_value=(self.manifest, self.lexical)):
            result = self.pipeline.retrieve('RAGAs 如何评估？', lexical_query='How does RAGAs evaluate?', collect_trace=True)
        self.lexical.search.assert_called_once_with('How does RAGAs evaluate?', 100, ['p1'])
        self.store.search.assert_called_once_with([1], 'snapshot', 100, ['p1'], False)
        self.assertTrue(self.models.rerank.call_args.args[1][0]['text'].startswith('Title:'))
        self.assertEqual(result['evidence'][0]['text'], 'body 0')
        self.assertEqual(result['route']['mode'], 'single_paper')
        self.assertEqual(result['bm25_query_source'], 'provided')
        self.assertEqual(len(result['rankings']['rrf20']), 6)

    def test_manual_scope_and_global_override_are_respected(self):
        alternate = [{**h, 'paper_id': 'p2'} for h in self.hits]
        self.lexical.search.return_value = self.store.search.return_value = alternate
        with patch.object(self.pipeline, 'snapshot', return_value=(self.manifest, self.lexical)):
            result = self.pipeline.retrieve('RAGAs 如何评估？', paper_ids=['p2'])
            self.assertEqual(result['route']['mode'], 'manual')
            global_result = self.pipeline.retrieve('RAGAs 如何评估？', auto_route=False)
            self.assertEqual(global_result['route']['mode'], 'global')
        self.assertIsNone(self.store.search.call_args.args[3])

    def test_ask_packs_all_reranked_candidates_without_generation(self):
        with patch.object(self.pipeline, 'snapshot', return_value=(self.manifest, self.lexical)), \
             patch('rag.pipeline.atomic_json'):
            result = self.pipeline.ask('RAGAs 如何评估？', generate_answer=False)
        self.assertEqual(len(result['evidence']), 6)
        self.assertLessEqual(result['context_chars'], 8000)
        self.assertEqual(result['generation_status'], 'disabled')

    def test_freeze_replays_identically_and_detects_code_changes(self):
        paths = ['rag/config.py', 'rag/pipeline.py', 'rag/retrieval.py', 'rag/routing.py',
                 'rag/embeddings.py', 'rag/generation.py', 'rag/benchmark.py', 'rag/frozen_baseline.py',
                 'scripts/accept_heldout.py', 'configs/paper_aliases.json']
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for path in paths:
                target = root/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture', encoding='utf-8')
            with patch('rag.frozen_baseline.current_manifest', return_value=self.manifest), \
                 patch('rag.frozen_baseline.PaperRouter.from_project', return_value=self.pipeline._router):
                first = freeze_baseline(self.cfg, root)
                self.assertEqual(first, freeze_baseline(self.cfg, root))
                (root/'rag/pipeline.py').write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, '偏离'):
                    freeze_baseline(self.cfg, root)

    def test_heldout_once_and_gold_ids_never_filter_pipeline(self):
        q = {'question_id': 'h1', 'question': 'RAGAs query?', 'question_en': 'English RAGAs query?',
             'answerable': True, 'source_scope': 'named_papers', 'gold_paper_ids': ['p1'],
             'gold_sources_exhaustive': True, 'evidence_units': [{'unit_id': 'u', 'alternatives': [[
                 {'paper_id': 'p1', 'document_sha256': 'sha', 'pdf_page': 1,
                  'char_start': 0, 'char_end': 4, 'quote': 'FACT'}]]}]}
        unsupported = {**q, 'question_id': 'h2', 'question': 'Unknown number?', 'answerable': False, 'evidence_units': []}
        hit = {**self.hits[0], 'text': 'FACT'}
        result = {'snapshot_id': 'snapshot', 'rankings': {'union': [hit], 'rrf20': [hit]},
                  'reranked_candidates': [hit], 'evidence': [hit]}
        pipeline = Mock()
        pipeline.snapshot.return_value = (self.manifest, self.lexical)
        pipeline.store.count.return_value = 6
        pipeline.retrieve.return_value = result
        frozen = {'spec_sha256': 'a'*64, 'spec': {'snapshot_id': 'snapshot'}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/'data/evaluation/paper_benchmark/heldout.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(q)+'\n'+json.dumps(unsupported), encoding='utf-8')
            with patch.object(accept_heldout, 'ROOT', root), \
                 patch.object(accept_heldout, 'load_settings', return_value=self.cfg), \
                 patch.object(accept_heldout, 'freeze_baseline', return_value=frozen), \
                 patch.object(accept_heldout, 'load_questions', return_value=[q, unsupported]), \
                 patch.object(accept_heldout, 'load_source_pages', return_value={('p1', 'sha', 1): 'FACT'}), \
                 patch.object(accept_heldout, 'Pipeline', return_value=pipeline) as factory, \
                 patch('sys.argv', ['accept_heldout', '--run']), redirect_stdout(io.StringIO()):
                accept_heldout.main()
                accept_heldout.main()
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(pipeline.retrieve.call_count, 2)
            for call in pipeline.retrieve.call_args_list:
                self.assertNotIn('paper_ids', call.kwargs)
                self.assertEqual(call.kwargs['lexical_query'], q['question_en'])
            target = root/'runtime/acceptance'/('heldout-'+'a'*12)
            summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual((summary['answerable'], summary['unsupported']), (1, 1))
            self.assertEqual(summary['groups'][0]['stages']['context_budget']['evidence_coverage'], 1)
            pipeline.store.close.assert_called_once()

    def test_generation_learning_reuses_cache_and_keeps_reference_out_of_prompt(self):
        q = {'question_id': 'q', 'question': 'Question?', 'question_en': 'Question?',
             'answerable': True, 'reference_answer': 'REFERENCE MUST NOT GO TO GENERATOR'}
        context = [{**self.hits[0], 'citation_id': 'S1'}]
        row = {'question': q['question'], 'budget_context': context, 'context_chars': 100}
        answer = {'answer': 'Grounded.[S1]', 'citations': ['S1'], 'insufficient_evidence': False}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root/'source'
            source.mkdir()
            (source/'q_single_query.json').write_text(json.dumps(row), encoding='utf-8')
            with patch.object(learn_generation, 'ROOT', root), \
                 patch.object(learn_generation, 'latest_development_cache', return_value=(source, {'status': 'completed', 'question_ids': ['q']})), \
                 patch.object(learn_generation, 'load_questions', return_value=[q]), \
                 patch.object(learn_generation, 'generation_settings', return_value=GenerationSettings(api_key='fixture-key')), \
                 patch.object(learn_generation, 'generate', return_value=answer) as generate, \
                 patch('sys.argv', ['learn_generation', '--generate']), redirect_stdout(io.StringIO()):
                learn_generation.main()
                learn_generation.main()
            generate.assert_called_once()
            self.assertEqual(generate.call_args.args[:2], (q['question'], context))
            preview, = (root/'runtime/generation_learning').glob('*/prompt_preview.json')
            content = preview.read_text(encoding='utf-8')
            self.assertNotIn('REFERENCE', content)
            self.assertNotIn('fixture-key', content)


if __name__ == '__main__':
    unittest.main()

"""Exercise the paired experiment without loading models or connecting to Qdrant."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from rag.config import RetrievalSettings
from scripts import compare_chunking as experiment


class ChunkingComparisonTests(unittest.TestCase):
    def run_fixture(self, root, *, fail_structure=False):
        fixed = RetrievalSettings(collection='fixed', batch_size=8)
        structure = RetrievalSettings(collection='structure', batch_size=16,
                                      chunk_strategy='layout-blocks', chunk_overlap=0)
        span = {'paper_id': 'p', 'document_sha256': 'sha', 'pdf_page': 1,
                'char_start': 2, 'char_end': 8, 'quote': 'cdefgh'}
        question = {'question_id': 'q1', 'question': '中文问题', 'question_en': 'English query',
                    'answerable': True, 'source_scope': 'named_papers', 'category': 'method',
                    'gold_paper_ids': ['p'], 'gold_sources_exhaustive': True,
                    'evidence_units': [{'unit_id': 'u', 'alternatives': [[span]]}]}
        questions = [question, {**question, 'question_id': 'q2',
                                'source_scope': 'open_corpus_anchor'}]
        target_hit = {**{k: span[k] for k in ('paper_id', 'document_sha256', 'pdf_page')},
                      'chunk_id': 'target', 'title': 'Paper', 'text': 'cdefgh'}
        distractors = [{'paper_id': 'noise', 'document_sha256': 'noise-sha', 'pdf_page': 1,
                        'chunk_id': f'noise-{i}', 'title': 'Other', 'text': f'unrelated {i}'}
                       for i in range(5)]
        structural_hit = {**target_hit, 'source_spans': [
            {'source_start': 2, 'source_end': 8, 'text_start': 0, 'text_end': 6}]}
        hits = {'fixed': [*distractors, target_hit], 'structure': [structural_hit, *distractors]}
        documents = {'p': 'sha', 'noise': 'noise-sha'}
        stores = {label: Mock() for label in hits}
        for label, store in stores.items():
            store.count.return_value = 6
            store.search.return_value = hits[label]
        if fail_structure:
            stores['structure'].validate_collection.side_effect = ValueError('bad collection')

        def current(cfg):
            return {'snapshot_id': cfg.collection, 'documents': documents,
                    'profile': cfg.index_profile(), 'chunks': 6}, hits[cfg.collection]

        def lexical(chunks):
            index = Mock()
            index.search.return_value = chunks
            return index

        models = Mock()
        models.query_vector.return_value = [1.0]
        # Keep RRF order to make target rank six in fixed and one in structure.
        models.rerank.side_effect = lambda query, candidates: candidates
        benchmark = root/'data/evaluation/paper_benchmark/dev.jsonl'
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text('\n'.join(json.dumps(q) for q in questions), encoding='utf-8')
        with patch.object(experiment, 'ROOT', root), \
             patch.object(experiment, 'load_questions', return_value=questions), \
             patch.object(experiment, 'load_settings', side_effect=[fixed, structure]), \
             patch.object(experiment, 'current_chunks', side_effect=current), \
             patch.object(experiment, 'QdrantStore', side_effect=lambda cfg: stores[cfg.collection]), \
             patch.object(experiment, 'LexicalIndex', side_effect=lexical), \
             patch.object(experiment, 'load_source_pages', return_value={('p', 'sha', 1): 'abcdefghij'}), \
             patch.object(experiment, 'LocalModels', return_value=models) as factory, \
             patch('sys.argv', ['compare_chunking', '--limit', '0']), redirect_stdout(io.StringIO()):
            if fail_structure:
                with self.assertRaisesRegex(ValueError, 'bad collection'):
                    experiment.main()
                factory.assert_not_called()
            else:
                experiment.main()
                factory.assert_called_once_with(fixed)
        for store in stores.values():
            store.close.assert_called_once()
        directory, = (root/'runtime/experiments').iterdir()
        return directory, models, stores

    def test_paired_pipeline_preserves_sources_and_uses_all_reranked_candidates(self):
        with tempfile.TemporaryDirectory() as name:
            directory, models, stores = self.run_fixture(Path(name))
            run = json.loads((directory/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(run['controls']['shared_model_settings']['batch_size'], 8)
            self.assertEqual(run['indexes']['structure']['settings']['batch_size'], 16)
            self.assertEqual(models.query_vector.call_count, 2)
            self.assertEqual(models.rerank.call_count, 4)
            self.assertTrue(models.rerank.call_args_list[0].args[1][0]['text'].startswith('Title:'))
            for label, store in stores.items():
                store.search.assert_called_with([1.0], label, top_k=100, exact=False)
            for label in ('fixed', 'structure'):
                result = json.loads((directory/f'q1_{label}.json').read_text(encoding='utf-8'))
                self.assertEqual(result['metrics']['after_top5']['evidence_unit_recall'],
                                 0 if label == 'fixed' else 1)
                self.assertEqual(result['metrics']['context_budget']['evidence_unit_recall'], 1)
                self.assertLessEqual(result['context_chars'], 8000)
                self.assertEqual(len(result['budget_context']), 6)
                target = next(hit for hit in result['budget_context'] if hit['chunk_id'] == 'target')
                self.assertEqual(target['text'], 'cdefgh')
            summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(len(summary['groups']), 6)
            self.assertEqual(summary['regressions_top5'], [])
            self.assertEqual(len(summary['paired_questions']), 2)
            self.assertEqual(summary['paired_questions'][0]['delta_structure_minus_fixed']['after_top5'], 1)

    def test_validation_failure_closes_clients_and_retains_failure_status(self):
        with tempfile.TemporaryDirectory() as name:
            directory, _, _ = self.run_fixture(Path(name), fail_structure=True)
            run = json.loads((directory/'run.json').read_text(encoding='utf-8'))
            self.assertEqual(run['status'], 'failed_partial_results_retained')
            self.assertFalse((directory/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from rag.config import RetrievalSettings
from rag.facet_reranking import fuse_facet_rankings, order_for_facet_coverage, facet_proxy
from rag.query_decomposition import QueryPlan, Subquery
from scripts import experiment_facet_rerank as experiment


class FacetRerankingTests(unittest.TestCase):
    def setUp(self):
        self.hits = {cid: {'chunk_id': cid, 'paper_id': 'p', 'text': f'body {cid}',
                          'score': 0, 'scores': {'rerank': 0}} for cid in 'abcdef'}
        self.rankings = {'original': [self.hits[c] for c in 'abcdef'],
                         'facet1': [self.hits[c] for c in 'bcfade'],
                         'facet2': [self.hits[c] for c in 'fedabc']}

    def test_fusion_uses_rank_positions_and_does_not_mutate_inputs(self):
        before = copy.deepcopy(self.rankings)
        fused = fuse_facet_rankings(self.rankings)
        a = next(h for h in fused if h['chunk_id'] == 'a')
        self.assertAlmostEqual(a['score'], .5/61 + .25/64 + .25/64)
        self.assertEqual(self.rankings, before)
        self.assertEqual({h['chunk_id'] for h in fused}, set('abcdef'))

    def test_selection_represents_unseen_facets_and_preserves_the_full_pool(self):
        fused = fuse_facet_rankings(self.rankings)
        ordered, steps = order_for_facet_coverage(self.rankings, fused)
        self.assertEqual(ordered[0]['chunk_id'], 'a')
        self.assertEqual(ordered[1]['chunk_id'], 'f')
        self.assertEqual(steps[1]['facets'], ['facet1', 'facet2'])
        self.assertTrue(all(facet_proxy({k: v for k, v in self.rankings.items() if k != 'original'}, ordered[:5]).values()))
        self.assertEqual(len(ordered), 6)
        self.assertEqual({h['chunk_id'] for h in ordered}, set('abcdef'))

    def test_changed_or_missing_candidates_fail(self):
        missing = copy.deepcopy(self.rankings)
        missing['facet1'].pop()
        with self.assertRaisesRegex(ValueError, 'Candidate set'):
            fuse_facet_rankings(missing)
        changed = {label: copy.deepcopy(hits) for label, hits in self.rankings.items()}
        changed['facet2'][0]['text'] = 'Title: synthetic prefix'
        with self.assertRaisesRegex(ValueError, 'bodies'):
            fuse_facet_rankings(changed)

    def test_no_facets_and_empty_candidates_are_safe(self):
        one = {'original': self.rankings['original']}
        fused = fuse_facet_rankings(one)
        ordered, steps = order_for_facet_coverage(one, fused)
        self.assertEqual(ordered, one['original'])
        self.assertEqual(steps, [])
        self.assertEqual(order_for_facet_coverage({'original': []}, []), ([], []))

    def test_rank_proxy_is_reported_independently_of_evidence_support(self):
        # The selector sees no annotated answers; a proxy hit is just top3 membership.
        facets = {'f': self.rankings['facet1']}
        self.assertEqual(facet_proxy(facets, [self.hits['a']]), {'f': False})
        self.assertEqual(facet_proxy(facets, [self.hits['b']]), {'f': True})

    def test_mocked_experiment_reuses_frozen_candidates_and_scores(self):
        cfg = RetrievalSettings()
        question = {'question_id': 'q', 'question': '论文p的完整步骤？', 'question_en': 'Steps in p?',
                    'answerable': True, 'source_scope': 'named_papers', 'category': 'method',
                    'gold_paper_ids': ['p'], 'gold_sources_exhaustive': True,
                    'evidence_units': [{'unit_id': 'u', 'alternatives': [[
                        {'paper_id': 'p', 'document_sha256': 'sha', 'pdf_page': 1,
                         'char_start': 0, 'char_end': 15, 'quote': 'TARGET EVIDENCE'}]]}]}
        texts = ['TARGET EVIDENCE', *[f'Distractor {i}.' for i in range(5)]]
        hits = [{'chunk_id': f'h{i}', 'paper_id': 'p', 'document_sha256': 'sha', 'pdf_page': 1,
                 'title': 'Paper p', 'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                 'score': 0, 'scores': {'rerank': 0}} for i, text in enumerate(texts)]
        original = [*hits[1:], hits[0]]
        plan = QueryPlan(subqueries=[Subquery(zh='论文p的细节步骤？', en='Detailed steps in p?')], reason='Detail')
        row = {'question_id': 'q', 'question': question['question'], 'scope': 'named_papers',
               'planner_cache': {'question_en': question['question_en'], 'plan': plan.model_dump()},
               'rankings': {'rrf20': original}, 'reranked_candidates': original,
               'metrics': {'rrf20': {'evidence_unit_recall': 1}},
               'route': {'mode': 'single_paper', 'paper_ids': ['p']}}
        models = Mock()
        models.rerank.side_effect = lambda query, candidates: list(reversed(candidates))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            benchmark = root/'data/evaluation/paper_benchmark/dev.jsonl'
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text(json.dumps(question), encoding='utf-8')
            source = root/'runtime/experiments/decomposition-20261006-100000-fixture'
            source.mkdir(parents=True)
            state = {'status': 'completed', 'split': 'dev', 'question_ids': ['q'],
                     'settings': cfg.model_dump(), 'snapshot_id': 'snapshot',
                     'benchmark_sha256': hashlib.sha256(benchmark.read_bytes()).hexdigest(),
                     'controls': {'rerank_k': 20, 'top_k': 5, 'rerank_input': 'title + body', 'context_char_budget': 8000}}
            (source/'run.json').write_text(json.dumps(state), encoding='utf-8')
            (source/'q_decomposed.json').write_text(json.dumps(row), encoding='utf-8')
            with patch.object(experiment, 'ROOT', root), \
                 patch.object(experiment, 'load_questions', return_value=[question]), \
                 patch.object(experiment, 'load_source_pages', return_value={('p', 'sha', 1): ' '.join(texts)}), \
                 patch.object(experiment, 'LocalModels', return_value=models), \
                 patch('sys.argv', ['experiment_facet_rerank', '--run', '--limit', '0']), redirect_stdout(io.StringIO()):
                experiment.main()
            self.assertEqual(models.rerank.call_count, 1)
            models.query_vector.assert_not_called()
            output, = (root/'runtime/experiments').glob('facet-rerank-*')
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['additional_model_pairs'], 6)
            self.assertEqual(summary['improvements_budget'], {'facet_rrf': [], 'facet_cover': []})
            for policy in ['facet_rrf', 'facet_cover']:
                self.assertEqual(summary['paired_changes'][policy][0]['delta_vs_original_rerank']['after_top5'], 1)
            self.assertEqual(json.loads((output/'run.json').read_text(encoding='utf-8'))['status'], 'completed')
            for policy in experiment.POLICIES:
                result = json.loads((output/f'q_{policy}.json').read_text(encoding='utf-8'))
                self.assertEqual(set(result['candidate_ids']), {h['chunk_id'] for h in hits})
                self.assertLessEqual(result['context_chars'], 8000)
                self.assertTrue(any(h['text'] == 'TARGET EVIDENCE' for h in result['reranked_candidates']))


if __name__ == '__main__':
    unittest.main()

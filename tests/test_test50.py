import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.config import RetrievalSettings
from scripts import accept_test50
from scripts.prepare_test50 import validate_rows

ROOT = Path(__file__).resolve().parents[1]


class Test50Checks(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT/'data/evaluation/paper_benchmark'
        self.raw = (self.directory/'test50.jsonl').read_text(encoding='utf-8')
        self.questions = [json.loads(s) for s in self.raw.splitlines() if s]
        self.metadata = json.loads((self.directory/'test50_manifest.json').read_text(encoding='utf-8'))

    def rows(self):
        old = set(self.metadata['cohorts']['previously_evaluated23'])
        return [{'question_id': q['question_id'], 'answerable': q['answerable'], 'scope': q['source_scope'],
                 'metrics': {s: {'evidence_unit_recall': (0 if q['question_id'] in old else 1) if q['answerable'] else None,
                                 'all_required_evidence': (q['question_id'] not in old) if q['answerable'] else None}
                             for s in accept_test50.STAGES}} for q in self.questions]

    def test_original_23_are_preserved_and_manifest_counts_match(self):
        old = [json.loads(s) for s in (self.directory/'heldout.jsonl').read_text(encoding='utf-8').splitlines() if s]
        self.assertEqual(self.questions[:23], old)
        self.assertEqual(len(self.questions), 50)
        self.assertEqual(sum(q['answerable'] for q in self.questions), 46)
        self.assertEqual(self.metadata['scopes']['open_corpus_anchor'], 3)
        self.assertEqual(hashlib.sha256(self.raw.encode()).hexdigest(), self.metadata['dataset_sha256'])
        accept_test50.validate_cohorts(self.questions, self.metadata)

    def test_overlapping_cohorts_are_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata['cohorts']['new_questions27'][0] = metadata['cohorts']['previously_evaluated23'][0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            accept_test50.validate_cohorts(self.questions, metadata)

    def test_old_and_new_statistics_remain_separate_and_exclude_unsupported(self):
        stats = accept_test50.cohort_summary(self.rows(), self.metadata)
        for name, answerable, unsupported, expected in [('previously_evaluated23', 21, 2, 0), ('new_questions27', 25, 2, 1)]:
            self.assertEqual((stats[name]['answerable'], stats[name]['unsupported']), (answerable, unsupported))
            group = next(g for g in stats[name]['groups'] if g['scope'] == 'all_answerable')
            self.assertEqual(group['stages']['context_budget']['evidence_coverage'], expected)
        total = next(g for g in stats['all50']['groups'] if g['scope'] == 'all_answerable')
        self.assertAlmostEqual(total['stages']['context_budget']['evidence_coverage'], 25/46)

    def test_invalid_source_quote_is_rejected(self):
        span = {'paper_id': 'p', 'document_sha256': 'sha', 'pdf_page': 1, 'char_start': 0,
                'char_end': 4, 'quote': 'FACT', 'quote_sha256': hashlib.sha256(b'FACT').hexdigest()}
        q = {'question_id': 'q', 'question': 'Q?', 'question_en': 'Q?', 'reference_answer': 'A',
             'gold_paper_ids': ['p'], 'answerable': True,
             'evidence_units': [{'unit_id': 'u', 'alternatives': [[span]]}],
             'required_answer_points': [{'text': 'A', 'evidence_unit_ids': ['u']}]}
        self.assertEqual(validate_rows([q], {('p', 'sha', 1): 'FACT'}, []), 1)
        span['char_end'] = 3
        with self.assertRaisesRegex(ValueError, 'quote/offset'):
            validate_rows([q], {('p', 'sha', 1): 'FACT'}, [])

    def test_completed_acceptance_replays_without_pdf_or_model_work(self):
        frozen = {'spec_sha256': 'f'*64}
        report = {'cohorts': accept_test50.cohort_summary(self.rows(), self.metadata)}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = root/'data/evaluation/paper_benchmark'
            data.mkdir(parents=True)
            (data/'test50.jsonl').write_bytes(self.raw.encode('utf-8'))
            (data/'test50_manifest.json').write_text(json.dumps(self.metadata), encoding='utf-8')
            target = root/'runtime/acceptance'/('test50-'+'f'*12+'-'+self.metadata['dataset_sha256'][:8])
            target.mkdir(parents=True)
            (target/'run.json').write_text(json.dumps({'status': 'completed'}), encoding='utf-8')
            (target/'summary.json').write_text(json.dumps(report), encoding='utf-8')
            with patch.object(accept_test50, 'ROOT', root), \
                 patch.object(accept_test50, 'load_settings', return_value=RetrievalSettings(candidate_k=100, context_char_budget=8000)), \
                 patch.object(accept_test50, 'freeze_baseline', return_value=frozen), \
                 patch.object(accept_test50, 'load_source_pages', side_effect=AssertionError('Unexpected PDF read')), \
                 patch.object(accept_test50, 'Pipeline', side_effect=AssertionError('Unexpected model/database work')), \
                 patch('sys.argv', ['accept_test50', '--run']), redirect_stdout(io.StringIO()):
                accept_test50.main()


if __name__ == '__main__':
    unittest.main()

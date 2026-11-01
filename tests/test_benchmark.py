"""Protect source-span scoring from overlap inflation and evidence leakage."""
import unittest

from rag.benchmark import interval_coverage, score_evidence


class EvidenceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.pages = {('p', 'sha', 1): 'abcdefghij', ('p', 'sha', 2): 'abcdefghij'}
        self.span = {'paper_id': 'p', 'document_sha256': 'sha', 'pdf_page': 1,
                     'char_start': 2, 'char_end': 8, 'quote': 'cdefgh'}
        self.question = {'question_id': 'q', 'answerable': True, 'gold_sources_exhaustive': True,
                         'evidence_units': [{'unit_id': 'u', 'alternatives': [[self.span]]}]}

    def chunk(self, text, page=1, digest='sha'):
        return {'paper_id': 'p', 'document_sha256': digest, 'pdf_page': page, 'text': text}

    def test_overlapping_chunks_are_unioned(self):
        self.assertEqual(interval_coverage(0, 10, [(0, 7), (3, 7)]), .7)
        result = score_evidence(self.question, [self.chunk('cdef'), self.chunk('efgh')], self.pages)
        self.assertTrue(result['all_required_evidence'])

    def test_missing_middle_is_not_complete(self):
        result = score_evidence(self.question, [self.chunk('cd'), self.chunk('gh')], self.pages)
        self.assertFalse(result['all_required_evidence'])

    def test_wrong_page_cannot_cover_evidence(self):
        self.assertEqual(score_evidence(self.question, [self.chunk('cdefgh', 2)], self.pages)['evidence_unit_recall'], 0)

    def test_changed_hash_fails_instead_of_false_score(self):
        with self.assertRaises(ValueError):
            score_evidence(self.question, [self.chunk('cdefgh', digest='changed')], self.pages)

    def test_or_alternatives_and_joint_spans(self):
        other = {**self.span, 'pdf_page': 2}
        self.question['evidence_units'][0]['alternatives'] = [[self.span, other]]
        self.assertFalse(score_evidence(self.question, [self.chunk('cdefgh')], self.pages)['all_required_evidence'])
        self.question['evidence_units'][0]['alternatives'].append([self.span])
        self.assertTrue(score_evidence(self.question, [self.chunk('cdefgh')], self.pages)['all_required_evidence'])

    def test_top_k_and_unsupported(self):
        self.assertFalse(score_evidence(self.question, [self.chunk('ab'), self.chunk('cdefgh')], self.pages, k=1)['all_required_evidence'])
        self.question['answerable'] = False
        self.question['evidence_units'] = []
        self.assertIsNone(score_evidence(self.question, [], self.pages)['evidence_unit_recall'])


if __name__ == '__main__':
    unittest.main()

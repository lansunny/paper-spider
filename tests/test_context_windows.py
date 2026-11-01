import unittest
from types import SimpleNamespace

from rag.context_windows import merge_source_windows, bounded_neighbor_windows


class ContextWindowTests(unittest.TestCase):
    def hit(self, text, cid, page=1, digest='sha'):
        return {'paper_id': 'p', 'document_sha256': digest, 'pdf_page': page,
                'text': text, 'chunk_id': cid, 'title': 'Paper'}

    def test_overlap_preserves_union_without_duplicate_text(self):
        pages = {('p', 'sha', 1): 'abcdefghij'}
        hits = [self.hit('cdefgh', 'one'), self.hit('abcdef', 'two'), self.hit('cdefgh', 'one')]
        windows = merge_source_windows(hits, pages)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]['text'], 'abcdefgh')
        self.assertEqual(windows[0]['source_chunk_ids'], ['one', 'two'])
        self.assertEqual(windows[0]['priority'], 0)

    def test_disjoint_ranges_do_not_fill_missing_text(self):
        pages = {('p', 'sha', 1): 'abcdefghij'}
        windows = merge_source_windows([self.hit('ij', 'a'), self.hit('ab', 'b')], pages)
        self.assertEqual([w['text'] for w in windows], ['ij', 'ab'])

    def test_pages_and_document_versions_stay_separate(self):
        pages = {('p', 'sha', 1): 'abcdef', ('p', 'sha', 2): 'abcdef', ('p', 'new', 1): 'abcdef'}
        windows = merge_source_windows([self.hit('abc', 'a'), self.hit('abc', 'b', page=2),
                                        self.hit('abc', 'c', digest='new')], pages)
        self.assertEqual(len(windows), 3)

    def test_ambiguous_source_text_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_source_windows([self.hit('abc', 'a')], {('p', 'sha', 1): 'abc abc'})

    def test_bounded_expansion_keeps_seed_and_respects_source_and_cap(self):
        pages = {('p', 'sha', 1): 'Alpha. Beta. Gamma. Delta.'}
        hits = [{**self.hit(text, str(i)), 'chunk_index': i} for i, text in enumerate(
            ['Alpha. Beta.', 'Beta.', 'Beta. Gamma. Delta.'])]
        chunks = [SimpleNamespace(**h, payload=lambda h=h: dict(h)) for h in hits]
        windows = bounded_neighbor_windows([hits[1]], chunks, pages, max_seed_chars=19)
        self.assertEqual(windows[0]['text'], 'Alpha. Beta. Gamma.')
        self.assertLessEqual(len(windows[0]['text']), 19)
        self.assertIn(hits[1]['text'], windows[0]['text'])
        merged = merge_source_windows(windows, pages)
        self.assertEqual(merged[0]['text'], windows[0]['text'])
        self.assertEqual(merged[0]['source_chunk_ids'], ['0', '1', '2'])

    def test_seed_is_never_silently_cut_to_meet_cap(self):
        seed = {**self.hit('Complete sentence.', 'a'), 'chunk_index': 0}
        with self.assertRaises(ValueError):
            bounded_neighbor_windows([seed], [], {('p', 'sha', 1): seed['text']}, max_seed_chars=4)


if __name__ == '__main__':
    unittest.main()

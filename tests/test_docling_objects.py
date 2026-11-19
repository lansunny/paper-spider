import re
import unittest

from rag.docling_objects import mentions, table_views, target_for_number, windows


class WordTokenizer:
    def encode(self, text, *, add_special_tokens, **kwargs):
        return [0] * (len(text.split()) + 2 * add_special_tokens)

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [m.span() for m in re.finditer(r'\S+', text)]}


class DoclingObjectsTest(unittest.TestCase):
    def test_single_paragraph_stays_whole_and_long_one_preserves_tail(self):
        text = '  one two three four five six seven eight nine ten eleven  '
        tokenizer = WordTokenizer()
        self.assertEqual(list(windows(text, tokenizer, 30, 2)), [(0, len(text))])
        slices = list(windows(text, tokenizer, 8, 2, 'Section\n'))
        self.assertEqual((slices[0][0], slices[-1][-1]), (0, len(text)))
        for (a, b), (c, d) in zip(slices, slices[1:]):
            self.assertLess(c, b)
            self.assertGreater(d, b)
            self.assertEqual(len(text[c:b].split()), 2)
            self.assertLessEqual(len(tokenizer.encode('Section\n' + text[a:b], add_special_tokens=True)), 8)

    def test_original_table_positions_drive_multilevel_headers(self):
        def cell(text, r, c, span=1, header=False):
            return dict(text=text, start_row_offset_idx=r, end_row_offset_idx=r+1,
                        start_col_offset_idx=c, end_col_offset_idx=c+span,
                        row_span=1, col_span=span, column_header=header)

        cells = [cell('NQ', 0, 0, 2, True), cell('EM', 1, 0, header=True),
                 cell('F1', 1, 1, header=True), cell('82.4', 2, 0), cell('91.7', 2, 1)]
        data = {'num_rows': 3, 'num_cols': 2, 'table_cells': cells}
        html, rows, columns = table_views(data)
        self.assertEqual(columns, ['NQ', 'EM', 'F1'])
        self.assertIn('colspan="2">NQ', html)
        self.assertEqual(rows, [{'row': 2, 'cell_ids': [3, 4], 'text': 'NQ / EM: 82.4 | NQ / F1: 91.7'}])
        self.assertEqual(table_views({**data, 'table_cells': cells + [cell('collision', 2, 0)]}), (None, [], []))

    def test_ambiguous_number_is_not_silently_resolved(self):
        text = 'See Figures 2a and 3, and Table A.1.'
        self.assertEqual([m[:2] for m in mentions(text)], [('figure', '2A'), ('figure', '3'), ('table', 'A.1')])
        figures = {'2': ['fig2'], '3': ['fig3', 'other3']}
        self.assertEqual(target_for_number('2A', lambda n: figures.get(n, [])), ['fig2'])
        self.assertEqual(target_for_number('3', lambda n: figures.get(n, [])), ['fig3', 'other3'])

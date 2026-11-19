import unittest

from scripts.evaluate_docling import score_cells, score_row


def cell(text, row, col, *, header=False):
    return dict(text=text, start_row_offset_idx=row, end_row_offset_idx=row+1,
                start_col_offset_idx=col, end_col_offset_idx=col+1, column_header=header)


class DoclingEvalTest(unittest.TestCase):
    def test_table_number_does_not_count_missing_header_as_correct(self):
        expected = [dict(row=6, col=1, value='0.72', header='NQ / C.R.')]
        cells = [cell('0.72', 6, 1)]
        header = [cell('NQ', 1, 1, header=True), cell('C.R.', 2, 1)]
        checks = score_cells(expected, cells, header)
        self.assertTrue(checks[0]['value_hit'])
        self.assertFalse(checks[0]['header_hit'])
        header[-1]['column_header'] = True
        self.assertTrue(score_cells(expected, cells, header)[0]['header_hit'])
        self.assertFalse(score_cells(expected, [cell('0.72', 6, 2)], header)[0]['value_hit'])

    def test_joint_requires_quote_range_and_figure(self):
        case = {'id':'case','group':'joint','gold':[dict(kind='paragraph',object_id='p',span=[2,12]),
                                                 dict(kind='figure',object_id='f')]}
        result = {'body':[dict(object_id='p',view_span=[0,7]),dict(object_id='p',view_span=[6,11])],
                  'figures':[dict(object_id='f')],'tables':[]}
        self.assertFalse(score_row(case,result,60)['all_evidence'])
        result['body'][1]['view_span'] = [6,13]
        self.assertTrue(score_row(case,result,60)['all_evidence'])

    def test_figure_ablation_uses_only_selected_branches(self):
        case = {'id':'case','group':'figure_text','gold':[dict(kind='figure',object_id='f')]}
        result = {'figures':[dict(object_id='f'),dict(object_id='wrong')],'body':[],'tables':[],
                  'rankings':{'figure':{'caption':[dict(object_id='wrong')],
                      'context':[dict(object_id='f',via='nearby_candidate')], 'clip':[dict(object_id='f')]}}}
        arms = score_row(case,result,60)['arms']
        self.assertFalse(arms['caption_explicit']['hit5'])
        self.assertTrue(arms['caption_context']['hit5'])
        self.assertTrue(arms['all']['hit1'])

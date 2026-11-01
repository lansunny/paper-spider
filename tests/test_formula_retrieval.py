import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from rag.formula_extraction import extract_document
from rag.formula_store import FormulaStore, publish_document
from rag.formula_retrieval import attach_formulas, aligned_spans
from rag.formula_evaluation import evaluate_formula_regression


class FormulaRetrievalTests(unittest.TestCase):
    def fixture(self, root, duplicate=False):
        path = root/'paper.pdf'
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((40, 50), 'An unrelated introductory paragraph.', fontsize=11)
            page.insert_text((40, 170), 'The formula combines the two variables.', fontsize=11)
            page.insert_text((180, 210), 'f = a + b', fontsize=14)
            page.insert_text((535, 210), '(1)', fontsize=14)
            page.insert_text((40, 460), 'An unrelated conclusion paragraph.', fontsize=11)
            page = pdf.new_page(width=600, height=800)
            page.insert_text((40, 100), 'Equation (1) defines the combined score.', fontsize=11)
            if duplicate:
                page.insert_text((180, 210), 'g = a + c', fontsize=14)
                page.insert_text((535, 210), '(1)', fontsize=14)
            pdf.save(path)
        doc = {'paper_id': 'p1', 'title': 'Fixture', '_path': path}
        output, elements, _ = extract_document(doc, root=root)
        publish_document(output, root)
        return doc, output, next(e for e in elements if e.equation_number == '1')

    def hit(self, formula, text, page=1, **more):
        return {'paper_id': 'p1', 'document_sha256': formula.document_sha256,
                'pdf_page': page, 'text': text, 'chunk_id': 'c1', **more}

    def test_full_authority_is_explicit_and_partial_run_cannot_replace_it(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); doc, full, _ = self.fixture(root)
            before = (root/'runtime/formulas/CURRENT.json').read_bytes()
            subset, _, _ = extract_document(doc, root=root, pages=[1])
            with self.assertRaises(ValueError):
                publish_document(subset, root)
            self.assertEqual(before, (root/'runtime/formulas/CURRENT.json').read_bytes())
            self.assertEqual(FormulaStore(root).document('p1')['record']['directory'], full.relative_to(root).as_posix())

    def test_body_join_uses_exact_shared_source_spans_and_marks_proxy_untrusted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root)
            s = next(s for s in e.source_spans if 'f = ' in s.text)
            span = {'block_number': s.block_number, 'source_start': s.source_start, 'source_end': s.source_end,
                    'text_start': 0, 'text_end': len(s.text)}
            hit = self.hit(e, s.text, source_spans=[span])
            result = attach_formulas('What is the formula?', [hit], root=root)
            f = result['formulas'][0]
            self.assertEqual(f['matches'][0]['method'], 'formula_inside_chunk')
            self.assertEqual(f['matches'][0]['alignment'], 'chunk_source_spans')
            self.assertEqual(f['proxy_trust'], 'unverified_pdf_text')
            self.assertFalse(f['model_has_seen_image'])
            self.assertFalse(result['formula_retrieval']['vector_index'])
            self.assertNotIn('formula_ids', hit)

    def test_legacy_unique_alignment_and_mismatch_are_distinguished(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root)
            result = attach_formulas('What is the formula?', [self.hit(e, 'f = a + b')], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['alignment'], 'legacy_unique_text_alignment')
            bad = self.hit(e, 'f = x + b', source_spans=[{'block_number': 2, 'source_start': 0, 'source_end': 9, 'text_start': 0, 'text_end': 9}])
            result = attach_formulas('Tell me about the method', [bad], root=root)
            self.assertEqual(result['formulas'], [])
            self.assertEqual(result['formula_retrieval']['diagnostics'][0]['reason'], 'source_coordinate_mismatch')
            page = {'text': 'x x', 'blocks': []}
            self.assertEqual(aligned_spans({'text': 'x'}, page)[1], 'legacy_text_missing_or_ambiguous')

    def test_disjoint_spans_do_not_attach_formula_in_the_gap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root)
            page = FormulaStore(root).document('p1')['pages'][1]
            blocks = [b for b in page['blocks'] if b['text'].startswith('An unrelated')]
            text = '\n\n'.join(b['text'] for b in blocks); cursor=0; spans=[]
            for b in blocks:
                spans.append({'block_number': b['number'], 'source_start': b['start'], 'source_end': b['end'],
                              'text_start': cursor, 'text_end': cursor+len(b['text'])})
                cursor += len(b['text'])+2
            result = attach_formulas('Tell me about the paper', [self.hit(e,text,source_spans=spans)], root=root)
            self.assertEqual(result['formulas'], [])

    def test_adjacent_context_and_cross_page_references_have_explicit_match_types(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root)
            result = attach_formulas('Explain the variables', [self.hit(e, 'The formula combines the two variables.')], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['method'], 'adjacent_explanation')
            result = attach_formulas('Explain the score', [self.hit(e,'Equation (1) defines the combined score.',page=2)], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['method'], 'explicit_formula_reference')

    def test_number_and_literal_symbol_lookup_do_not_require_vectors_or_ocr(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); self.fixture(root)
            result = attach_formulas('公式 (1)', [], paper_ids=['p1'], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['method'], 'exact_equation_number')
            result = attach_formulas('f = a + b', [], paper_ids=['p1'], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['method'], 'literal_symbol_proxy_fallback')
            result = attach_formulas('公式 f = a + b 是什么意思？', [], paper_ids=['p1'], root=root)
            self.assertEqual(result['formulas'][0]['matches'][0]['method'], 'literal_symbol_proxy_fallback')
            self.assertEqual(attach_formulas('(1)', [], paper_ids=['p1'], root=root)['formulas'], [])

    def test_duplicate_numbers_and_stale_document_hash_never_choose_an_arbitrary_formula(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root, duplicate=True)
            result = attach_formulas('公式 (1)', [], paper_ids=['p1'], root=root)
            self.assertEqual(result['formulas'], [])
            self.assertEqual(result['formula_retrieval']['diagnostics'][0]['reason'], 'ambiguous_equation_number')
            self.assertIsNone(FormulaStore(root).document('p1','f'*64))
            stale = {**self.hit(e,'f = a + b'), 'document_sha256': 'f'*64}
            self.assertEqual(attach_formulas('Explain the method',[stale],root=root)['formulas'], [])

    def test_api_returns_original_crop_and_rejects_changed_image_or_inactive_ids(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); _, _, e = self.fixture(root)
            path = root/e.image_path; before=path.read_bytes()
            with patch('backend.app.ROOT', root):
                client=TestClient(app); url=f'/api/formulas/p1/{e.element_id}/image'
                response=client.get(url)
                self.assertEqual(response.status_code,200)
                self.assertEqual(hashlib.sha256(response.content).hexdigest(),e.image_sha256)
                self.assertEqual(client.get('/api/formulas/p1/no-such-id/image').status_code,404)
                path.write_bytes(b'changed')
                self.assertEqual(client.get(url).status_code,409)
            self.assertNotEqual(before,path.read_bytes())

    def test_gold_source_anchors_survive_changed_extraction_ids(self):
        import json
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); doc,_,e=self.fixture(root)
            s=e.source_spans[0]
            annotation={'annotation_id':'fixed_gold_1','paper_id':'p1','document_sha256':e.document_sha256,
                        'pdf_page':1,'bbox':list(e.bbox),'label':'formula','equation_number':'1',
                        'canonical_page_sha256':s.canonical_page_sha256,'source_start':s.source_start,
                        'source_end':s.source_end,'source_quote':s.text,'group':'gold','review_basis':'original source'}
            path=root/'labels.json'; path.write_text(json.dumps({'version':'fixture','annotations':[annotation]}),encoding='utf-8')
            before=evaluate_formula_regression(path,root)
            with patch('rag.formula_extraction.VERSION','test-next-version'), patch('rag.formula_extraction.IDENTITY_VERSION','test-next-id'):
                output,_,_=extract_document(doc,root=root); publish_document(output,root)
            after=evaluate_formula_regression(path,root)
            self.assertTrue(before['all_passed'] and after['all_passed'])
            self.assertNotEqual(before['results'][0]['matched_element_ids'],after['results'][0]['matched_element_ids'])


if __name__ == '__main__':
    unittest.main()

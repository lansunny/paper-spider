import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag.formula_extraction import extract_document, import_ocr_results, page_records, explicit_references, region_from_group, math_font


class FormulaExtractionTests(unittest.TestCase):
    def make_pdf(self, root, mode='fraction'):
        path = root/'fixture.pdf'
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            if mode == 'two_columns':
                for x in [40, 330]:
                    for y in [70, 85]:
                        page.insert_text((x, y), 'This paragraph explains several mathematical methods.', fontsize=8)
                page.insert_text((90, 190), 'f = x + y', fontsize=14)
                page.insert_text((270, 190), '(1)', fontsize=14)
                page.insert_text((390, 190), 'g = z + t', fontsize=14)
                page.insert_text((555, 190), '(2)', fontsize=14)
            elif mode == 'caption':
                page.insert_text((70, 100), 'Figure 1: f = x + y (3)', fontsize=12)
            elif mode == 'scanned':
                pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
                pix.clear_with(255)
                page.insert_image(pymupdf.Rect(70, 90, 530, 650), pixmap=pix)
            elif mode == 'raster':
                pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 20), False)
                pix.clear_with(255)
                page.insert_image(pymupdf.Rect(200, 140, 470, 175), pixmap=pix)
                page.insert_text((535, 160), '(3)', fontsize=12)
            elif mode == 'function_argument':
                page.insert_text((180, 130), 'f = g(1)', fontsize=14)
            else:
                page.insert_text((40, 90), 'The score combines numerator and denominator.', fontsize=11)
                page.insert_text((180, 130), 'f = a + b', fontsize=14)
                page.draw_line((219, 134), (277, 134))
                page.insert_text((240, 148), 'n', fontsize=14)
                if mode != 'unnumbered':
                    page.insert_text((535, 130), '(1)', fontsize=14)
                page.insert_text((40, 210), 'Equation (1) defines the score, where n is the count.', fontsize=11)
                if mode == 'rotated':
                    page.set_rotation(90)
            pdf.save(path)
        return {'paper_id': 'fixture', 'title': 'Fixture paper', 'landing_url': 'https://example.org/fixture', '_path': path}

    def test_fraction_merges_fragments_links_context_and_preserves_pdf(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            doc = self.make_pdf(root)
            before = hashlib.sha256(doc['_path'].read_bytes()).hexdigest()
            _, elements, manifest = extract_document(doc, root=root)
            formula = next(e for e in elements if e.equation_number == '1')
            self.assertIn('n', formula.raw_text)
            self.assertGreater(formula.bbox[3], 147)
            self.assertTrue(formula.references)
            self.assertIn('n is the count', ' '.join(c.text for c in formula.context))
            self.assertTrue((root/formula.image_path).exists())
            self.assertEqual(before, hashlib.sha256(doc['_path'].read_bytes()).hexdigest())
            self.assertFalse(manifest['latex_recognition_performed'])
            with pymupdf.open(doc['_path']) as pdf:
                canonical = page_records(pdf[0])[0]
            for c in formula.context + formula.references:
                self.assertEqual(canonical[c.source_start:c.source_end], c.text)
                self.assertEqual(c.canonical_page_sha256, hashlib.sha256(canonical.encode()).hexdigest())

    def test_two_columns_do_not_join_equations(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, elements, _ = extract_document(self.make_pdf(root, 'two_columns'), root=root)
            left = next(e for e in elements if e.equation_number == '1')
            right = next(e for e in elements if e.equation_number == '2')
            self.assertLess(left.bbox[2], 300)
            self.assertGreater(right.bbox[0], 300)
            self.assertNotIn('g =', left.raw_text)
            self.assertNotIn('f =', right.raw_text)

    def test_captions_are_excluded_and_function_argument_is_not_a_number(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, elements, _ = extract_document(self.make_pdf(root, 'caption'), root=root)
            self.assertEqual(elements, [])
            _, elements, _ = extract_document(self.make_pdf(root, 'function_argument'), root=root)
            self.assertTrue(elements)
            self.assertTrue(all(e.equation_number is None for e in elements))

    def test_unnumbered_scanned_and_raster_states_are_explicit(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, elements, _ = extract_document(self.make_pdf(root, 'unnumbered'), root=root)
            self.assertTrue(elements)
            self.assertTrue(all(e.equation_number is None and e.review_status == 'unreviewed' for e in elements))
            _, elements, manifest = extract_document(self.make_pdf(root, 'scanned'), root=root)
            self.assertEqual(elements, [])
            self.assertEqual(manifest['pages_needing_ocr_layout'], [1])
            output, elements, _ = extract_document(self.make_pdf(root, 'raster'), root=root)
            self.assertEqual(elements, [])  # An arbitrary picture plus a number is not proof of a formula.
            pending = json.loads((output/'manifest.json').read_text(encoding='utf-8'))['pending_layout']
            self.assertEqual(pending[0]['reason'], 'equation_body_unresolved')
            self.assertEqual(pending[0]['evidence'], 'nearby_image_not_proof_of_formula')
            self.assertTrue((output/'ocr_jobs.jsonl').exists())

    def test_rotation_cache_and_resolution_do_not_corrupt_source_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            doc = self.make_pdf(root, 'rotated')
            _, first, _ = extract_document(doc, root=root, dpi=180)
            _, second, cached = extract_document(doc, root=root, dpi=180)
            self.assertTrue(cached['cache_hit'])
            self.assertEqual(first, second)
            self.assertEqual(first[0].page_rotation, 90)
            self.assertEqual(first[0].page_width, 600)
            _, other_resolution, _ = extract_document(doc, root=root, dpi=240)
            self.assertEqual([e.element_id for e in first], [e.element_id for e in other_resolution])
            self.assertNotEqual(first[0].image_sha256, other_resolution[0].image_sha256)
            (root/first[0].image_path).write_bytes(b'corrupted')
            _, repaired, state = extract_document(doc, root=root, dpi=180)
            self.assertFalse(state['cache_hit'])
            self.assertEqual(first[0].image_sha256, repaired[0].image_sha256)

    def test_ocr_import_rejects_stale_crops_and_cannot_claim_verification(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, elements, manifest = extract_document(self.make_pdf(root), root=root)
            e = next(e for e in elements if e.equation_number == '1')
            result = {'element_id': e.element_id, 'document_sha256': e.document_sha256,
                      'image_sha256': e.image_sha256, 'engine': 'fixture-ocr', 'engine_revision': 'r1',
                      'latex': r'f=\frac{a+b}{n}'}
            before = (output/'elements.jsonl').read_bytes()
            self.assertEqual(import_ocr_results(output, [result], root), 1)
            self.assertEqual(before, (output/'elements.jsonl').read_bytes())
            recognized = json.loads((output/'recognitions.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(recognized['status'], 'recognized_unverified')
            with self.assertRaises(ValueError):
                import_ocr_results(output, [{**result, 'image_sha256': 'f'*64}], root)
            with self.assertRaises(ValueError):
                import_ocr_results(output, [{**result, 'status': 'verified'}], root)
            self.assertEqual(manifest['elements'], len(elements))

    def test_long_reference_excerpt_contains_mention_and_keeps_exact_offsets(self):
        text = 'word ' * 400 + 'Equation (12) explains the loss. ' + 'tail ' * 100
        block = {'number': 0, 'text': text, 'start': 0, 'end': len(text), 'bbox': (0, 0, 100, 100)}
        refs = explicit_references('12', {1: (text, [block])})
        self.assertEqual(len(refs), 1)
        self.assertIn('Equation (12)', refs[0].text)
        self.assertEqual(refs[0].text, text[refs[0].source_start:refs[0].source_end])
        self.assertEqual(explicit_references('1', {1: (text, [block])}), [])

    def test_glyphs_recovered_from_prose_keep_their_source_line_identity(self):
        core = {'id': 'b1:l0', 'block': 1, 'bbox': (10, 10, 80, 22), 'body_bbox': (10, 10, 80, 22),
                'body_text': 'f = a', 'text': 'f = a', 'caption': False, 'number': None,
                'chars': [{'c': c, 'font': 'CMMI10', 'bbox': (10, 10, 80, 22)} for c in 'f=a']}
        body = 'b, where the variable describes the denominator in this expression'
        prose = {'id': 'b2:l0', 'block': 2, 'bbox': (62, 21, 270, 34), 'body_bbox': (62, 21, 270, 34),
                 'body_text': body, 'text': body, 'caption': False, 'number': None,
                 'chars': [{'c': 'b', 'font': 'CMMI10', 'bbox': (62, 21, 70, 34)}] +
                          [{'c': c, 'font': 'Helvetica', 'bbox': (80, 21, 270, 34)} for c in body[1:]]}
        result = region_from_group([core], [core, prose])
        self.assertEqual({line['id'] for line in result['lines']}, {'b1:l0', 'b2:l0'})
        self.assertEqual(result['bbox'][3], 34)

    def test_math_fonts_include_bold_italic_and_ams_without_treating_code_as_math(self):
        for font in ['ABCDEF+CMMIB10', 'MSBM10', 'MSAM10', 'LMMath-Regular', 'MTMI', 'EulerMath', 'rsfs10']:
            self.assertTrue(math_font(font), font)
        self.assertFalse(math_font('Inconsolatazi4-Regular'))

    def test_code_urls_years_and_right_aligned_prose_are_not_equations(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/'negatives.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=600, height=800)
                page.insert_text((240, 80), 'Kadavath et al. (2022)', fontsize=11)
                page.insert_text((130, 120), 'This sentence has several interpretations (1)', fontsize=11)
                page.insert_text((220, 170), "result = evaluate(amnesty_qa['eval'])", fontname='cour', fontsize=11)
                page.insert_text((260, 210), 'abs/2304.13734.', fontsize=11)
                page.insert_text((240, 250), 'BM25 + DPR', fontsize=11)
                pdf.save(path)
            _, elements, manifest = extract_document({'_path': path, 'paper_id': 'neg', 'title': 'Negatives'}, root=root)
            self.assertEqual(elements, [])
            self.assertEqual(manifest['pending_layout'], [])

    def test_numbered_fraction_with_english_words_keeps_complete_numerator(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root/'word_fraction.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=600, height=800)
                page.insert_text((110, 200), 'CR =', fontsize=12)
                page.insert_text((150, 190), 'number of extracted sentences', fontsize=10)
                page.draw_line((148, 195), (325, 195))
                page.insert_text((150, 207), 'total number of sentences', fontsize=10)
                page.insert_text((530, 200), '(2)', fontsize=12)
                pdf.save(path)
            _, elements, _ = extract_document({'_path': path, 'paper_id': 'words', 'title': 'Words'}, root=root)
            formula = next(e for e in elements if e.equation_number == '2')
            self.assertIn('number of extracted sentences', formula.raw_text)
            self.assertIn('total number of sentences', formula.raw_text)

    def test_caption_associated_table_and_vector_figure_are_excluded_locally(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); path=root/'layout.pdf'
            with pymupdf.open() as pdf:
                p=pdf.new_page(width=600,height=800)
                for y in [60,160]:
                    p.draw_line((320,y),(570,y))
                p.insert_text((330,100),'G.+BM25(1)',fontsize=11)
                p.insert_text((320,185),'Table 1: Comparison of the methods.',fontsize=11)
                p.draw_rect((45,240,265,320),fill=(.9,.9,.9))
                p.insert_text((65,270),'x = y + z',fontsize=11)
                p.insert_text((45,345),'Figure 1: An illustration of the process.',fontsize=11)
                p.insert_text((180,450),'f = a + b',fontsize=14)
                p.insert_text((535,450),'(1)',fontsize=14)
                pdf.save(path)
            _,elements,_=extract_document({'_path':path,'paper_id':'layout','title':'Layout'},root=root)
            self.assertEqual(len(elements),1)
            self.assertEqual(elements[0].equation_number,'1')


if __name__ == '__main__':
    unittest.main()

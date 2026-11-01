import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag.benchmark import chunk_intervals, interval_coverage
from rag.chunking import normalized_blocks
from rag.config import RetrievalSettings, load_settings
from rag.ingestion import current_chunks, current_manifest, pdf_pages, fingerprint, build_index
from rag.structural_chunking import structural_chunks, ordered_blocks


class WordTokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
        return {'input_ids': list(range(len(offsets))), 'offset_mapping': offsets}


def block(n, text, x0, y0, x1, y1):
    return {'block_number': n, 'text': text, 'bbox': [x0, y0, x1, y1]}


class StructuralTests(unittest.TestCase):
    def config(self):
        return RetrievalSettings(chunk_strategy='layout-blocks', chunk_size=32, chunk_overlap=0,
                                 embedding_max_length=64, embedding_revision='test')

    def test_two_columns_with_full_width_barrier_and_exact_provenance(self):
        t = 'Sentence about retrieval and evidence. ' * 3
        page = {'pdf_page': 1, 'page_width': 600, 'blocks': [
            block(0, t+'Right B', 320, 80, 570, 110),
            block(1, t+'Left A', 30, 30, 280, 60),
            block(2, t+'Right A', 320, 30, 570, 60),
            block(3, t+'Left B', 30, 80, 280, 110),
            block(4, '2 Results', 30, 150, 570, 165),
            block(5, t+'Right C', 320, 180, 570, 210),
            block(6, t+'Left C', 30, 180, 280, 210)]}
        canonical, order = ordered_blocks(page)
        self.assertEqual([r['number'] for r in order], [1, 3, 2, 0, 4, 6, 5])
        chunks = list(structural_chunks(page, WordTokenizer(), self.config(), paper_id='p', document_sha256='a'*64))
        key = ('p', 'a'*64, 1)
        intervals = chunk_intervals(chunks, {key: canonical})[key]
        self.assertEqual(interval_coverage(0, len(canonical), intervals), 1)
        columns = {r['number']: (r['column'], r['region']) for r in order}
        for c in chunks:
            self.assertEqual({columns[s.block_number] for s in c.source_spans}, {(c.layout_column, c.layout_region)})

    def test_oversized_sentence_fallback_preserves_text_and_bounds(self):
        text = ' '.join(f'word{i}' for i in range(90))
        page = {'pdf_page': 2, 'page_width': 600, 'blocks': [block(0, text, 30, 30, 570, 400)]}
        chunks = list(structural_chunks(page, WordTokenizer(), self.config(), paper_id='p', document_sha256='a'*64))
        self.assertTrue(any(c.split_method == 'token-fallback' for c in chunks))
        self.assertTrue(all(len(c.text.split()) <= 32 for c in chunks))
        key = ('p', 'a'*64, 2)
        self.assertEqual(interval_coverage(0, len(text), chunk_intervals(chunks, {key: text})[key]), 1)

    def test_source_spans_reject_forged_text(self):
        page = {'pdf_page': 1, 'page_width': 600, 'blocks': [block(0, 'Original text.', 30, 30, 200, 50)]}
        chunk = next(structural_chunks(page, WordTokenizer(), self.config(), paper_id='p', document_sha256='a'*64)).payload()
        chunk['text'] = 'Modified text.'
        with self.assertRaises(ValueError):
            chunk_intervals([chunk], {('p', 'a'*64, 1): 'Original text.'})

    def test_merged_column_blocks_can_be_noncontiguous_in_original_text(self):
        text = 'retrieval evidence annotation sentences ' * 3
        page = {'pdf_page': 1, 'page_width': 600, 'blocks': [
            block(0, 'LEFT A '+text, 30, 20, 280, 50),
            block(1, 'RIGHT A '+text, 320, 20, 570, 50),
            block(2, 'LEFT B '+text, 30, 55, 280, 85),
            block(3, 'RIGHT B '+text, 320, 55, 570, 85)]}
        canonical, _ = normalized_blocks(page['blocks'])
        chunks = list(structural_chunks(page, WordTokenizer(), self.config(), paper_id='p', document_sha256='a'*64))
        left = next(c for c in chunks if c.layout_column == 'left')
        self.assertEqual([s.block_number for s in left.source_spans], [0, 2])
        self.assertNotIn(left.text, canonical)
        key = ('p', 'a'*64, 1)
        ranges = chunk_intervals([left], {key: canonical})[key]
        self.assertEqual(len(ranges), 2)
        self.assertFalse(any(canonical.index('RIGHT A') >= a and canonical.index('RIGHT A') < b for a, b in ranges))

    def test_structural_build_and_reload_without_real_models(self):
        from test_rag import FakeModels, MemoryStore
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data').mkdir()
            pdf_path = root/'fixture.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page().insert_textbox((50, 50, 500, 500), 'Sentence with source evidence. '*40)
                pdf.save(pdf_path)
            (root/'data/paper_manifest.json').write_text(json.dumps([{
                'paper_id': 'p', 'title': 'Fixture', 'status': 'downloaded', 'pdf_path': 'fixture.pdf'}]), encoding='utf8')
            cfg = self.config().model_copy(update={'vector_size': 3, 'schema_version': 2})
            built = build_index(cfg, FakeModels(), MemoryStore(), root)
            loaded, chunks = current_chunks(cfg, root)
            self.assertEqual(built['snapshot_id'], loaded['snapshot_id'])
            self.assertTrue(all(c.source_spans for c in chunks))
            self.assertEqual(loaded['profile']['embedding_input'], 'body-v1')

    def test_corrupt_and_truncated_cache_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'fixture.pdf'
            with pymupdf.open() as pdf:
                for text in ['First page.', 'Second page.']:
                    pdf.new_page().insert_text((50, 50), text)
                pdf.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            document = {'paper_id': 'fixture', 'sha256': digest, '_path': path}
            cache = root/'runtime/parsed'/f'{digest}.jsonl'
            cache.parent.mkdir(parents=True)
            cache.write_text('{"broken":', encoding='utf8')
            rows = pdf_pages(document, root)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]['page_width'] > 0)
            cache.write_text(json.dumps(rows[0])+'\n', encoding='utf8')
            self.assertEqual(len(pdf_pages(document, root)), 2)
            self.assertEqual(len(list((cache.parent/'quarantine').glob('*.jsonl'))), 2)

    def test_existing_snapshot_stays_compatible(self):
        # Exercise a prebuilt snapshot when available; synthetic indexes are tested above.
        try:
            cfg = load_settings()
        except FileNotFoundError:
            self.skipTest('Local config unavailable')
        if current_manifest(cfg) is None:
            self.skipTest('Optional prebuilt local snapshot unavailable')
        manifest, chunks = current_chunks(cfg)
        self.assertEqual(manifest['chunk_ids_hash'], fingerprint([c.chunk_id for c in chunks]))
        self.assertEqual(cfg.index_profile(), manifest['profile'])


if __name__ == '__main__':
    unittest.main()

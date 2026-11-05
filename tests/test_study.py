"""Reading persistence, source isolation and real PDF annotation export."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pymupdf

from backend.study import create_study_router


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf = self.root / 'paper.pdf'
        with pymupdf.open() as document:
            document.new_page(width=500, height=700).insert_text((70, 90), 'Research question and evidence.')
            second = document.new_page(width=500, height=700)
            second.insert_text((70, 90), 'Methods and results.')
            second.set_rotation(90)
            document.set_toc([[1, 'Introduction', 1], [1, 'Methods', 2]])
            document.save(self.pdf)
        self.digest = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.paper = {'paper_id': 'paper', 'title': 'A research paper', 'pdf_path': 'paper.pdf', 'sha256': self.digest}
        app = FastAPI()
        app.include_router(create_study_router(self.root, lambda identifier: self.paper if identifier == 'paper' else None))
        self.client = TestClient(app)
        self.base = '/api/study/papers/paper'

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def patch(self, **fields):
        return self.client.patch(self.base + '/state', json={'document_sha256': self.digest, **fields})

    def mark(self, kind='highlight', page=1):
        annotation = {'id': str(uuid4()), 'kind': kind, 'page': page, 'color': '#74a9f5',
                      'rects': [[.1, .1, .4, .15]], 'quote': 'Research question', 'text': 'A useful observation'}
        if kind == 'ink':
            annotation.update(rects=[], points=[[.1, .3], [.2, .32], [.3, .3]])
        return annotation

    def put(self, annotation, digest=None):
        return self.client.put(self.base + '/annotations/' + annotation['id'], json={
            'document_sha256': digest or self.digest, 'annotation': annotation})

    def test_metadata_does_not_require_rag_or_a_vector_database(self):
        data = self.client.get(self.base).json()
        self.assertEqual(data['pages'], 2)
        self.assertEqual(data['document_sha256'], self.digest)
        self.assertEqual(data['outline'][1]['page'], 2)
        self.assertNotIn('pdf_path', data['paper'])

    def test_notes_and_progress_are_independent_and_persist(self):
        self.assertEqual(self.patch(notes='My own note', notes_revision=0).status_code, 200)
        result = self.patch(page=2, zoom='125', scroll_position=.45, bookmarks=[2, 2, 1]).json()
        self.assertEqual(result['notes'], 'My own note')
        self.assertEqual(result['notes_revision'], 1)
        self.assertEqual(result['bookmarks'], [1, 2])
        self.assertAlmostEqual(result['scroll_position'], .45)
        self.assertEqual(self.client.get(self.base).json()['state']['page'], 2)

    def test_stale_note_write_is_rejected_without_losing_the_saved_note(self):
        self.patch(notes='From first window', notes_revision=0)
        response = self.patch(notes='From stale window', notes_revision=0, page=2)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()['detail']['notes_revision'], 1)
        state = self.client.get(self.base).json()['state']
        self.assertEqual(state['notes'], 'From first window')
        self.assertEqual(state['page'], 1)

    def test_annotation_put_is_idempotent_and_deletion_is_persistent(self):
        annotation = self.mark()
        first = self.put(annotation).json()
        annotation['text'] = 'Edited comment'
        second = self.put(annotation).json()
        self.assertEqual(first['created_at'], second['created_at'])
        marks = self.client.get(self.base).json()['state']['annotations']
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]['text'], 'Edited comment')
        self.assertTrue(self.client.delete(self.base + '/annotations/' + annotation['id'], params={'document_sha256': self.digest}).json()['deleted'])
        self.assertEqual(self.client.get(self.base).json()['state']['annotations'], [])

    def test_invalid_pages_and_coordinates_are_rejected(self):
        self.assertEqual(self.patch(page=3).status_code, 422)
        self.assertEqual(self.patch(bookmarks=[0]).status_code, 422)
        self.assertEqual(self.patch(scroll_position=2).status_code, 422)
        annotation = self.mark(page=3)
        self.assertEqual(self.put(annotation).status_code, 422)
        annotation.update(page=1, rects=[[-.1, 0, .4, .3]])
        self.assertEqual(self.put(annotation).status_code, 422)
        annotation.update(rects=[[.4, 0, .1, .3]])
        self.assertEqual(self.put(annotation).status_code, 422)

    def test_different_pdf_versions_do_not_share_annotations_or_notes(self):
        self.patch(notes='Original version', notes_revision=0)
        self.put(self.mark())
        with pymupdf.open(self.pdf) as source:
            source[0].insert_text((70, 140), 'An updated document.')
            new = self.root / 'new.pdf'
            source.save(new)
        new.replace(self.pdf)
        self.assertEqual(self.patch(page=2).status_code, 409)
        current = self.client.get(self.base).json()
        self.assertNotEqual(current['document_sha256'], self.digest)
        self.assertEqual(current['state']['notes'], '')
        self.assertEqual(current['state']['annotations'], [])

    def test_export_creates_real_pdf_annotations_and_keeps_original_unchanged(self):
        for kind in ('highlight', 'underline', 'ink', 'rectangle', 'note'):
            self.assertEqual(self.put(self.mark(kind)).status_code, 200)
        self.assertEqual(self.put(self.mark('highlight', page=2)).status_code, 200)
        response = self.client.get(self.base + '/export', params={'format': 'pdf', 'document_sha256': self.digest})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/pdf')
        with pymupdf.open(stream=response.content, filetype='pdf') as result:
            self.assertEqual(len(list(result[0].annots())), 5)
            self.assertEqual(len(list(result[1].annots())), 1)
            self.assertEqual(result[1].rotation, 90)
        self.assertEqual(hashlib.sha256(self.pdf.read_bytes()).hexdigest(), self.digest)

    def test_markdown_and_json_export_keep_notes_and_source_identity(self):
        self.patch(notes='## My insight\nA note.', notes_revision=0)
        self.put(self.mark())
        md = self.client.get(self.base + '/export', params={'format': 'md', 'document_sha256': self.digest})
        self.assertIn('## My insight', md.text)
        self.assertIn('Research question', md.text)
        data = self.client.get(self.base + '/export', params={'format': 'json', 'document_sha256': self.digest}).json()
        self.assertEqual(data['document_sha256'], self.digest)
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(len(data['annotations']), 1)

    def test_unknown_papers_and_paths_outside_the_workspace_are_rejected(self):
        self.assertEqual(self.client.get('/api/study/papers/missing').status_code, 404)
        self.paper['pdf_path'] = '../outside.pdf'
        self.assertEqual(self.client.get(self.base).status_code, 404)

    def test_recent_readings_include_real_progress_and_counts_without_note_content(self):
        self.assertEqual(self.client.get('/api/study/papers/recent').json(), {'papers': []})
        self.patch(page=2, notes='A private reading note', notes_revision=0)
        self.put(self.mark())
        result = self.client.get('/api/study/papers/recent')
        self.assertEqual(result.status_code, 200)
        recent = result.json()['papers']
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]['paper_id'], 'paper')
        self.assertEqual(recent[0]['page'], 2)
        self.assertEqual(recent[0]['pages'], 2)
        self.assertEqual(recent[0]['annotation_count'], 1)
        self.assertEqual(recent[0]['notes_characters'], len('A private reading note'))
        self.assertNotIn('notes', recent[0])
        self.assertNotIn('pdf_path', recent[0])

    def test_recent_readings_exclude_previous_pdf_versions(self):
        self.patch(notes='Old version note', notes_revision=0)
        with pymupdf.open(self.pdf) as source:
            source[0].insert_text((70, 170), 'New PDF version.')
            replacement = self.root / 'replacement.pdf'
            source.save(replacement)
        replacement.replace(self.pdf)
        self.assertEqual(self.client.get('/api/study/papers/recent').json()['papers'], [])
        self.client.get(self.base)
        recent = self.client.get('/api/study/papers/recent').json()['papers']
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]['notes_characters'], 0)
        self.assertNotEqual(recent[0]['document_sha256'], self.digest)


if __name__ == '__main__':
    unittest.main()

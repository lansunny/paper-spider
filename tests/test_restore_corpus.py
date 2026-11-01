import hashlib
from pathlib import Path
import tempfile
import unittest

import httpx

from scripts.restore_paper_corpus import restore_paper


class RestoreCorpusTests(unittest.TestCase):
    def paper(self,content):
        return {'paper_id':'2024.test.1','pdf_url':'https://aclanthology.org/2024.test.1.pdf',
                'sha256':hashlib.sha256(content).hexdigest()}

    def test_verified_pdf_is_cached_without_repeating_network_work(self):
        data=b'%PDF-fixture'
        calls=[]
        with tempfile.TemporaryDirectory() as d:
            with httpx.Client(transport=httpx.MockTransport(lambda r:(calls.append(r.url) or httpx.Response(200,content=data)))) as client:
                self.assertEqual(restore_paper(self.paper(data),Path(d),client)['status'],'downloaded')
                self.assertEqual(restore_paper(self.paper(data),Path(d),client)['status'],'cached')
            self.assertEqual(len(calls),1)

    def test_changed_source_does_not_become_an_active_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b'%PDF-changed'))) as client:
                with self.assertRaisesRegex(ValueError,'Source PDF changed'):
                    restore_paper(self.paper(b'%PDF-original'),root,client)
            self.assertEqual(list((root/'data/pdfs').iterdir()),[])

    def test_existing_different_file_and_unsafe_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'data/pdfs/2024.test.1.pdf';target.parent.mkdir(parents=True)
            target.write_bytes(b'keep existing content')
            with httpx.Client(transport=httpx.MockTransport(lambda r:self.fail('Unexpected network call'))) as client:
                with self.assertRaisesRegex(ValueError,'Existing PDF differs'):
                    restore_paper(self.paper(b'%PDF-original'),root,client)
                bad={**self.paper(b'%PDF-original'),'paper_id':'../../escape'}
                with self.assertRaisesRegex(ValueError,'Invalid catalog paper ID'):
                    restore_paper(bad,root,client)
            self.assertEqual(target.read_bytes(),b'keep existing content')


if __name__=='__main__': unittest.main()

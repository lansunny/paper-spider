from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_github_release import prepare_release,public_files


class PublicReleaseTests(unittest.TestCase):
    def fixture(self,root):
        files={'README.md':'# Paper Spider\n','main.py':'print("Paper Spider")\n',
               '.env.example':'DEEPSEEK_API_KEY=\n','rag/pipeline.py':'value = 1\n',
               '.env':'private configuration','runtime/models/weights.bin':'private weights',
               'resume/private.md':'private notes','docs/学习导航.md':'private learning notes',
               'data/pdfs/paper.pdf':'private PDF','vendor/reference/README.md':'reference only'}
        for name,text in files.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

    def test_release_is_allowlisted_and_preserves_private_source_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            target,archive,manifest,report=prepare_release(root,root/'runtime/github-preview/review')
            self.assertEqual(report['files'],4)
            self.assertFalse((target/'.env').exists())
            self.assertFalse((target/'runtime').exists())
            self.assertFalse((target/'data/pdfs').exists())
            self.assertEqual((root/'.env').read_text(),'private configuration')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'paper-spider/README.md','paper-spider/main.py','paper-spider/.env.example','paper-spider/rag/pipeline.py'})
            self.assertTrue(manifest.is_file())

    def test_credential_like_content_stops_export_before_writing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'rag/pipeline.py').write_text('secret = '+repr('sk-'+'x'*40))
            with self.assertRaisesRegex(ValueError,'Credential-like'):
                prepare_release(root,root/'runtime/github-preview/review')
            self.assertFalse((root/'runtime/github-preview/review').exists())

    def test_output_cannot_escape_review_directory_or_overwrite_an_existing_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaisesRegex(ValueError,'under runtime/github-preview'):
                prepare_release(root,root/'elsewhere')
            prepare_release(root,root/'runtime/github-preview/review')
            with self.assertRaisesRegex(ValueError,'already exists'):
                prepare_release(root,root/'runtime/github-preview/review')


if __name__=='__main__': unittest.main()

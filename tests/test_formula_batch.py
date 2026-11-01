import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from rag.formula_batch import run_formula_batch, select_papers, batch_state_path
from rag.formula_extraction import extract_document
from rag.formula_store import FormulaStore


class FormulaBatchTests(unittest.TestCase):
    def paper(self, root, pid, expression='f = a + b'):
        path = root/'data'/f'{pid}.pdf'
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=600, height=800)
            page.insert_text((40, 80), 'The formula describes a combination of two values.', fontsize=11)
            page.insert_text((180, 140), expression, fontsize=14)
            page.insert_text((535, 140), '(1)', fontsize=14)
            page = pdf.new_page(width=600, height=800)
            page.insert_text((40, 80), 'Equation (1) provides the combination.', fontsize=11)
            pdf.save(path)
        return {'paper_id': pid, 'title': 'Fixture '+pid,
                'pdf_path': path.relative_to(root).as_posix(), 'landing_url': 'https://example.org/'+pid}

    def read_state(self, root):
        return json.loads((root/'runtime/formula_batch/progress.json').read_text(encoding='utf-8'))

    def test_complete_batch_publishes_all_and_rerun_validates_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); papers=[self.paper(root,'a'),self.paper(root,'b')]
            first=run_formula_batch(papers,root=root)
            self.assertEqual((first['completed'],first['failed'],first['cached']),(2,0,0))
            self.assertEqual(first['pages'],4)
            active=FormulaStore(root); self.assertEqual(set(active.records),{'a','b'})
            crop=active.image('a',active.document('a')['elements'][0].element_id)
            before=crop.stat().st_mtime_ns
            second=run_formula_batch(papers,root=root)
            self.assertEqual((second['completed'],second['cached']),(2,2))
            self.assertEqual(before,crop.stat().st_mtime_ns)
            self.assertEqual(self.read_state(root)['last_run']['status'],'completed')

    def test_one_failure_does_not_abort_and_retry_selects_only_failed_papers(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            missing={'paper_id':'missing','title':'Missing','pdf_path':'data/missing.pdf'}
            papers=[self.paper(root,'a'),missing,self.paper(root,'b')]
            result=run_formula_batch(papers,root=root)
            self.assertEqual((result['completed'],result['failed'],result['processed']),(2,1,3))
            self.assertEqual(result['status'],'completed_with_failures')
            self.assertEqual(result['failed_ids'],['missing'])
            self.assertEqual(set(FormulaStore(root).records),{'a','b'})
            self.paper(root,'missing')
            result=run_formula_batch(papers,root=root,retry_failed=True)
            self.assertEqual(result['requested_ids'],['missing'])
            self.assertEqual((result['completed'],result['failed']),(1,0))
            self.assertEqual(set(FormulaStore(root).records),{'a','b','missing'})
            result=run_formula_batch(papers,root=root,retry_failed=True)
            self.assertEqual(result['total'],0)

    def test_interrupt_checkpoints_current_paper_and_rerun_continues(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); papers=[self.paper(root,'a'),self.paper(root,'b')]
            def interrupted(document,**kwargs):
                if document['paper_id']=='b':
                    raise KeyboardInterrupt
                return extract_document(document,**kwargs)
            with patch('rag.formula_batch.extract_document',side_effect=interrupted):
                result=run_formula_batch(papers,root=root)
            self.assertEqual(result['status'],'interrupted')
            state=self.read_state(root)
            self.assertEqual(state['papers']['a']['status'],'completed')
            self.assertEqual(state['papers']['b']['status'],'interrupted')
            self.assertEqual(set(FormulaStore(root).records),{'a'})
            resumed=run_formula_batch(papers,root=root)
            self.assertEqual((resumed['completed'],resumed['cached'],resumed['failed']),(2,1,0))

    def test_completed_flag_does_not_hide_a_changed_pdf_or_dpi(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); paper=self.paper(root,'a')
            run_formula_batch([paper],root=root)
            old=FormulaStore(root).records['a']['document_sha256']
            paper=self.paper(root,'a','g = x + y')
            result=run_formula_batch([paper],root=root)
            self.assertEqual(result['cached'],0)
            self.assertNotEqual(old,FormulaStore(root).records['a']['document_sha256'])
            result=run_formula_batch([paper],root=root,dpi=240)
            self.assertEqual(result['cached'],0)
            self.assertEqual(FormulaStore(root).document('a')['elements'][0].render_dpi,240)

    def test_corrupt_crop_is_repaired_instead_of_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); paper=self.paper(root,'a'); run_formula_batch([paper],root=root)
            store=FormulaStore(root); e=store.document('a')['elements'][0]; path=store.image('a',e.element_id)
            original=path.read_bytes(); path.write_bytes(b'broken')
            result=run_formula_batch([paper],root=root)
            self.assertEqual(result['cached'],0)
            self.assertEqual(path.read_bytes(),original)

    def test_subset_selection_is_explicit_and_checkpoint_cannot_overwrite_other_artifacts(self):
        papers=[{'paper_id':'a'},{'paper_id':'b'}]
        self.assertEqual(select_papers(papers,limit=1),papers[:1])
        self.assertEqual(select_papers(papers,['b','b']),papers[1:])
        with self.assertRaises(ValueError): select_papers(papers,['unknown'])
        with self.assertRaises(ValueError): select_papers(papers,limit=-1)
        with self.assertRaises(ValueError): select_papers(papers+papers[:1])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with self.assertRaises(ValueError): batch_state_path(root,root/'runtime/formulas/CURRENT.json')
            with self.assertRaises(ValueError): batch_state_path(root,root/'.env')

    def test_progress_reports_pages_and_cached_state_without_changing_extraction(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); paper=self.paper(root,'a'); document={**paper,'_path':root/paper['pdf_path']}
            events=[]
            _,first,_=extract_document(document,root=root,progress=lambda *event: events.append(event))
            self.assertEqual([e for e in events if e[0]=='detect'],[('detect',1,2),('detect',2,2)])
            self.assertEqual([e for e in events if e[0]=='render'],[('render',1,2),('render',2,2)])
            events=[]
            _,second,cached=extract_document(document,root=root,progress=lambda *event: events.append(event))
            self.assertTrue(cached['cache_hit']); self.assertEqual(first,second)
            self.assertIn(('cache_hit',2,2),events)
            self.assertFalse(any(e[0]=='detect' for e in events))


if __name__=='__main__':
    unittest.main()

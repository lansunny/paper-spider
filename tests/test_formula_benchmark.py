import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag.config import RetrievalSettings
from rag.formula_benchmark import (FormulaQuestion,prepare_dataset,load_dataset,formula_matches,score_formulas,sha)
from rag.formula_extraction import extract_document
from rag.formula_store import FormulaStore,publish_document
from rag.formula_experiment import evaluate_question,preflight,run_experiment,summarize
from rag.ingestion import index_directory
from rag.storage import atomic_json


class FormulaBenchmarkTests(unittest.TestCase):
    def fixture(self,root):
        gold_sources=[];hits=[];documents={}
        for pid,symbol in [('a','f'),('b','g')]:
            source=root/(pid+'.pdf')
            with pymupdf.open() as pdf:
                p=pdf.new_page(width=600,height=800)
                p.insert_text((40,80),'This paragraph explains the formula.',fontsize=11)
                p.insert_text((180,150),symbol+' = x + y',fontsize=14)
                p.insert_text((535,150),'(1)',fontsize=14)
                pdf.save(source)
            with pymupdf.open(source) as pdf:
                boxes=[b[:4] for b in pdf[0].get_text('blocks') if symbol+' = ' in b[4] or b[4].strip()=='(1)']
                rect=pymupdf.Rect(boxes[0])
                for box in boxes[1:]:rect|=pymupdf.Rect(box)
            digest=sha(source.read_bytes());documents[pid]=digest
            item={'gold_id':'gold_'+pid,'paper_id':pid,'title':'Paper '+pid,'document_sha256':digest,
                  'pdf_path':source.name,'source_url':'https://example.org/'+pid,'pdf_page':1,
                  'equation_number':'1','bbox':list(rect),'reference_answer':symbol+' is the sum of x and y.'}
            gold_sources.append(item)
            directory,_,_=extract_document({'paper_id':pid,'title':'Paper '+pid,'_path':source},root=root)
            publish_document(directory,root)
            hits.append({'paper_id':pid,'document_sha256':digest,'pdf_page':1,'text':symbol+' = x + y',
                         'chunk_id':'chunk_'+pid,'title':'Paper '+pid,'score':1.0,'scores':{'bm25':1.0}})
        questions=[{'question_id':'q1','question':'展示相加的原公式','question_en':'Show the addition formula.',
                    'category':'semantic_open','gold_groups':[['gold_a','gold_b']],
                    'reference_answer':'Either of the reviewed source formulas is acceptable.'},
                   {'question_id':'q2','question':'同时展示两篇论文中的原公式','question_en':'Show both paper formulas.',
                    'category':'semantic_cross','gold_groups':[['gold_a'],['gold_b']],
                    'reference_answer':'Both original formulas are required.'}]
        data=root/'data/evaluation/formula_test50';data.mkdir(parents=True)
        atomic_json(data/'curation.json',{'version':'fixture','gold_sources':gold_sources,'questions':questions})
        prepare_dataset(data,root)
        questions,gold,meta=load_dataset(data,root)
        settings=RetrievalSettings(embedding_revision='fixture',vector_size=3,context_char_budget=2000)
        manifest={'profile':settings.index_profile(),'snapshot_id':'fixture-snapshot','documents':documents,'chunks':2}
        index=index_directory(settings,root);index.mkdir(parents=True)
        atomic_json(index/'CURRENT.json',manifest)
        (index/'fixture-snapshot.jsonl').write_text('\n'.join(json.dumps(h) for h in hits),encoding='utf-8')
        return questions,gold,meta,settings,manifest,hits

    def prediction(self,gold,gid,method='formula_inside_chunk',eid='any-extraction-id'):
        g=gold[gid]
        return {'element_id':eid,'paper_id':g.paper_id,'document_sha256':g.document_sha256,
                'pdf_page':g.pdf_page,'equation_number':g.equation_number,'bbox':g.bbox,
                'matches':[{'method':method,'source_chunk_id':'x'}]}

    def test_native_labels_are_independent_of_extraction_ids_and_alternatives_use_or(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);questions,gold,_,_,_,_=self.fixture(root)
            one=self.prediction(gold,'gold_a',eid='changed-id-after-new-algorithm')
            self.assertTrue(score_formulas(questions[0],[one],gold)['all_required'])
            multi=score_formulas(questions[1],[one],gold)
            self.assertEqual(multi['anchor_recall'],.5);self.assertFalse(multi['all_required'])
            self.assertTrue(score_formulas(questions[1],[one,self.prediction(gold,'gold_b')],gold)['all_required'])
            self.assertFalse(formula_matches({**one,'document_sha256':'f'*64},gold['gold_a']))
            self.assertFalse(formula_matches({**one,'pdf_page':2},gold['gold_a']))
            self.assertFalse(formula_matches({**one,'equation_number':'2'},gold['gold_a']))

    def test_number_lookup_does_not_count_as_text_join_recall(self):
        with tempfile.TemporaryDirectory() as name:
            questions,gold,_,_,_,_=self.fixture(Path(name))
            f=self.prediction(gold,'gold_a',method='exact_equation_number')
            self.assertEqual(score_formulas(questions[0],[f],gold)['anchor_recall'],1)
            self.assertEqual(score_formulas(questions[0],[f],gold,joins_only=True)['anchor_recall'],0)

    def test_controls_distinguish_false_attachment_from_number_safety(self):
        with tempfile.TemporaryDirectory() as name:
            _,gold,_,_,_,_=self.fixture(Path(name));f=self.prediction(gold,'gold_a')
            q=FormulaQuestion(question_id='neg',question='Python code',question_en='Python code',category='control',
                              expectation='no_formula',reference_answer='Not a mathematical formula.')
            self.assertTrue(score_formulas(q,[],gold)['control_passed'])
            self.assertFalse(score_formulas(q,[f],gold)['control_passed'])
            q=q.model_copy(update={'expectation':'no_exact_number','requested_equation_number':'9999'})
            self.assertTrue(score_formulas(q,[f],gold)['control_passed'])
            self.assertFalse(score_formulas(q,[{**f,'equation_number':'9999'}],gold)['control_passed'])

    def test_source_image_and_annotation_changes_fail_integrity_checks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);_,gold,_,_,_,_=self.fixture(root)
            path=root/gold['gold_a'].image_path;path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'crop changed'):
                load_dataset(root/'data/evaluation/formula_test50',root)

    def fake_pipeline(self,root,settings,manifest,hits,calls):
        class Store:
            def validate_collection(self):pass
            def count(self,snapshot):return manifest['chunks']
            def close(self):pass
        class Pipeline:
            def __init__(self):
                self.root=root;self.settings=settings;self.store=Store()
            def snapshot(self):return manifest,object()
            def retrieve(self,question,**kwargs):
                calls.append((question,kwargs))
                return {'snapshot_id':manifest['snapshot_id'],'route':{'mode':'global','paper_ids':[]},
                        'evidence':hits[:5],'reranked_candidates':hits,'timings':{},'candidate_counts':{'merged':len(hits)},
                        'rankings':{s:hits for s in ['bm25','dense','union','rrf20','after_top5']}}
        return Pipeline()

    def test_real_attachment_is_scored_and_gold_is_never_passed_to_retrieval(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);questions,gold,_,settings,manifest,hits=self.fixture(root)
            calls=[];pipeline=self.fake_pipeline(root,settings,manifest,hits,calls);store=FormulaStore(root)
            _,available=preflight(settings,gold,root,store)
            row=evaluate_question(questions[1],gold,pipeline,store,manifest,available)
            self.assertTrue(row['metrics']['context_budget']['all_required'])
            self.assertEqual(row['failure_diagnosis'],[])
            question,kwargs=calls[0]
            self.assertEqual(question,questions[1].question)
            self.assertNotIn('paper_ids',kwargs)
            self.assertEqual(set(kwargs),{'mode','top_k','rerank','lexical_query','auto_route','collect_trace'})
            self.assertEqual(kwargs['lexical_query'],questions[1].question_en)

    def test_rerank_loss_is_reported_without_pretending_the_formula_was_absent(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);questions,gold,_,settings,manifest,hits=self.fixture(root)
            calls=[];pipeline=self.fake_pipeline(root,settings,manifest,hits,calls)
            original=pipeline.retrieve
            def pruned(*args,**kwargs):
                r=original(*args,**kwargs);r['reranked_candidates']=hits[:1];r['rankings']['after_top5']=hits[:1];return r
            pipeline.retrieve=pruned;store=FormulaStore(root);_,available=preflight(settings,gold,root,store)
            row=evaluate_question(questions[1],gold,pipeline,store,manifest,available)
            self.assertEqual(row['metrics']['union']['anchor_recall'],1)
            self.assertEqual(row['metrics']['context_budget']['anchor_recall'],.5)
            self.assertEqual(row['failure_diagnosis'][0]['reason'],'rerank_or_context_budget_lost_formula_link')

    def test_semantic_and_lookup_summaries_are_separate(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);questions,gold,_,settings,manifest,hits=self.fixture(root)
            store=FormulaStore(root);_,available=preflight(settings,gold,root,store)
            row=evaluate_question(questions[0],gold,self.fake_pipeline(root,settings,manifest,hits,[]),store,manifest,available)
            other={**row,'question_id':'lookup','category':'exact_number'}
            summary=summarize([row,other])
            self.assertEqual(summary['semantic_main']['questions'],1)
            self.assertEqual(summary['lookup_diagnostics']['questions'],1)

    def test_cached_run_does_not_repeat_retrieval_and_changed_profile_gets_new_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);questions,gold,meta,settings,manifest,hits=self.fixture(root)
            names=['rag/pipeline.py','rag/retrieval.py','rag/routing.py','rag/embeddings.py','rag/vector_store.py',
                   'rag/generation.py','rag/formula_retrieval.py','rag/formula_store.py','rag/formula_benchmark.py',
                   'rag/formula_experiment.py','configs/paper_aliases.json']
            for filename in names:
                p=root/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture',encoding='utf-8')
            calls=[];factory=lambda *_:self.fake_pipeline(root,settings,manifest,hits,calls)
            target,_=run_experiment(questions,gold,meta,settings,root,pipeline_factory=factory)
            self.assertEqual(len(calls),2)
            reused,_=run_experiment(questions,gold,meta,settings,root,pipeline_factory=lambda *_:self.fail('Cached run loaded models'))
            self.assertEqual(target,reused)
            changed,_=run_experiment(questions,gold,meta,settings,root,english_lexical=False,pipeline_factory=factory)
            self.assertNotEqual(target,changed)
            self.assertEqual(len(calls),4)


if __name__=='__main__':unittest.main()

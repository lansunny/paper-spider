"""Graph scope, version guards, candidate budgets and unchanged V2 behavior."""
import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rag.config import RetrievalSettings
from rag.formula_store import digest_file
from rag.paper_graph.retrieval import GraphRetrievalOptions, augment_candidates, plan_expansion
from rag.paper_graph.store import PaperGraphStore, publish
from rag.pipeline_graph import GraphAugmentedPipeline
from rag.pipeline_v2 import PipelineV2


def hit(identifier, paper='a'):
    return {'chunk_id': identifier, 'paper_id': paper, 'title': 'Paper ' + paper,
            'text': 'Original body for ' + identifier, 'document_sha256': paper*64,
            'pdf_page': 1, 'score': 1., 'scores': {}}


class GraphRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        directory = self.root / 'runtime/paper_graph/builds/fixture'
        directory.mkdir(parents=True)
        with sqlite3.connect(directory / 'graph.sqlite3') as c:
            c.execute('CREATE TABLE papers(paper_id TEXT,title TEXT,document_sha256 TEXT,profile_sha256 TEXT)')
            c.execute('CREATE TABLE edges(dimension TEXT,source TEXT,target TEXT,weight REAL)')
            for pid in 'abcd':
                c.execute('INSERT INTO papers VALUES(?,?,?,?)', (pid, 'Paper '+pid, pid*64, 'profile-'+pid))
            c.executemany('INSERT INTO edges VALUES(?,?,?,?)', [('method','a','b',.9), ('method','a','c',.8),
                ('topic','a','d',.95), ('method','b','d',.7)])
        c.close()
        publish(self.root, directory, {'status':'completed','build_id':'graph-fixture',
            'source_snapshot_id':'structure-snapshot', 'artifacts':{'graph.sqlite3':digest_file(directory/'graph.sqlite3')}})
        self.context = {'question':'比较这些方法', 'lexical_query':'compare methods', 'mode':'hybrid', 'exact':False,
            'manifest':{'snapshot_id':'fixed-snapshot','documents':{pid:pid*64 for pid in 'abcd'}},
            'route':{'mode':'global','matched_paper_ids':[]},'paper_ids':None,
            'query_vectors':{'比较这些方法':'cached-vector'},'facets':[], 'protected_chunk_ids':[], 'rerank_total':4}
        self.options = GraphRetrievalOptions(enabled=True,dimension='method',seed_limit=1,candidate_slots=2)
        self.candidates = [hit('a'+str(i)) for i in range(4)]

    def tearDown(self):
        self.temp.cleanup()

    def pipeline(self, cls=GraphAugmentedPipeline):
        calls=[];rows=[hit('a'+str(i)) for i in range(8)]
        class Models:
            def query_vector(self,q): calls.append(('embed',q));return 'cached-vector'
            def rerank(self,q,hits): calls.append(('rerank',len(hits)));return [{**h,'score':10-i} for i,h in enumerate(hits)]
        class Store:
            def validate_collection(self): pass
            def search(self,vector,snapshot,top_k,paper_ids,exact):
                calls.append(('dense',vector,paper_ids))
                return [hit(p+'0',p) for p in paper_ids] if paper_ids else rows[:top_k]
        class Lexical:
            def query(self,q): return {'text':q,'tokens':[]}
            def search(self,q,top_k,paper_ids=None):
                return [hit(p+'1',p) for p in paper_ids] if paper_ids else rows[:top_k]
        pipeline=cls(RetrievalSettings(candidate_k=8,rerank_k=4,context_char_budget=1000),
            root=self.root,models=Models(),store=Store())
        pipeline.snapshot=lambda:(self.context['manifest'],Lexical())
        return pipeline,calls

    def test_disabled_requests_follow_the_same_v2_path(self):
        baseline,_=self.pipeline(PipelineV2);graph,_=self.pipeline()
        with patch('rag.pipeline_v2.perf_counter',return_value=1):
            a=baseline.retrieve('问题',auto_route=False)
            b=graph.retrieve('问题',auto_route=False)
        self.assertEqual(a,b)
        self.assertNotIn('graph_retrieval',b)

    def test_one_hop_uses_pdf_hashes_across_different_chunk_snapshots(self):
        info=plan_expansion(PaperGraphStore(self.root),self.candidates,self.context,self.options)
        self.assertEqual(info['expanded_paper_ids'],['b','c'])
        self.assertEqual(info['graph_source_snapshot_id'],'structure-snapshot')
        self.assertEqual(info['retrieval_snapshot_id'],'fixed-snapshot')
        self.assertNotIn('d',info['expanded_paper_ids'])

    def test_single_paper_scope_never_expands(self):
        for route,scope in [({'mode':'single_paper'},None),({'mode':'manual'},['a'])]:
            context=self.context|{'route':route,'paper_ids':scope}
            info=plan_expansion(PaperGraphStore(self.root),self.candidates,context,self.options)
            self.assertEqual(info['reason'],'single_paper_scope')
            self.assertEqual(info['expanded_paper_ids'],[])

    def test_manual_scope_and_stale_pdf_versions_are_respected(self):
        context=copy.deepcopy(self.context);context['paper_ids']=['a','b']
        info=plan_expansion(PaperGraphStore(self.root),self.candidates,context,self.options)
        self.assertEqual(info['expanded_paper_ids'],['b'])
        context=copy.deepcopy(self.context);context['manifest']['documents']['b']='changed'
        info=plan_expansion(PaperGraphStore(self.root),self.candidates,context,self.options)
        self.assertEqual(info['expanded_paper_ids'],['c'])

    def test_explicit_paper_seeds_are_not_mixed_with_unrelated_retrieved_papers(self):
        context=self.context|{'route':{'mode':'global','matched_paper_ids':['b']}}
        info=plan_expansion(PaperGraphStore(self.root),self.candidates,context,
            self.options.model_copy(update={'seed_limit':2}))
        self.assertEqual([s['paper_id'] for s in info['seeds']],['b'])
        self.assertEqual(info['seed_selection'],'explicit_papers_only')

    def test_graph_additions_keep_fixed_rerank_budget_and_reuse_query_vector(self):
        pipeline,calls=self.pipeline()
        result=pipeline.retrieve('比较这些方法',auto_route=False,top_k=3,
            graph_retrieval=self.options.model_dump())
        self.assertEqual(result['candidate_counts']['rerank_candidates'],4)
        self.assertEqual(result['candidate_counts']['graph_added'],2)
        self.assertEqual(sum(c[1] for c in calls if c[0]=='rerank'),4)
        self.assertEqual(sum(c[0]=='embed' for c in calls),1)
        for row in result['reranked_candidates']:
            self.assertEqual(row['text'],'Original body for '+row['chunk_id'])
        self.assertFalse(result['graph_retrieval']['summary_used_as_evidence'])

    def test_formula_source_candidates_are_protected(self):
        pipeline,_=self.pipeline();_,lexical=pipeline.snapshot()
        context=self.context|{'lexical':lexical,'protected_chunk_ids':['a3']}
        rows,meta=augment_candidates(pipeline,self.candidates,context,self.options)
        self.assertIn('a3',[r['chunk_id'] for r in rows])
        self.assertEqual(len(rows),4)

    def test_missing_graph_falls_back_without_losing_baseline_evidence(self):
        (self.root/'runtime/paper_graph/CURRENT.json').unlink()
        pipeline,_=self.pipeline();_,lexical=pipeline.snapshot()
        rows,meta=augment_candidates(pipeline,self.candidates,self.context|{'lexical':lexical},self.options)
        self.assertEqual(rows,self.candidates)
        self.assertEqual(meta['graph_retrieval']['reason'],'graph_not_built')

    def test_ask_context_uses_only_body_evidence_under_the_same_budget(self):
        pipeline,_=self.pipeline()
        result=pipeline.ask('比较这些方法',mode='hybrid',top_k=3,auto_route=False,
            generate_answer=False,graph_retrieval=self.options.model_dump())
        self.assertLessEqual(result['context_chars'],1000)
        self.assertTrue(result['graph_retrieval']['context_graph_chunk_ids'])
        self.assertTrue(all(h['text'].startswith('Original body') for h in result['evidence']))


if __name__=='__main__':unittest.main()

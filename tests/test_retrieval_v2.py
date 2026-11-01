import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.config import ROOT, RetrievalSettings
from rag.formula_retrieval_v2 import attach_formulas_v2
from rag.formula_retrieval_v2 import rescue_formula_sources
from rag.lexical_v2 import LexicalIndexV2, lexical_tokens
from rag.pipeline_v2 import PipelineV2, RetrievalPolicyV2
from rag.query_planning_v2 import PaperRouterV2, balanced_candidates, paper_facets
from rag.schema import Chunk


def hit(cid, pid='a', text='Original body'):
    return {'chunk_id': cid, 'paper_id': pid, 'title': pid, 'text': text,
            'document_sha256': 'a'*64, 'pdf_page': 1, 'scores': {}, 'score': 1}


class V2PlanningTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{'paper_id':'a','title':'Alpha: A Scientific Method'},
                        {'paper_id':'b','title':'Beta: Another Scientific Method'}]
        self.router = PaperRouterV2(self.catalog, ['a','b'], {'a':['DPR'], 'b':['RocketQA']})

    def test_per_paper_facets_derive_only_from_unambiguous_question_mentions(self):
        q='对比 DPR 与 RocketQA 的训练损失，分母是什么？请分别展示原公式。'
        facets=paper_facets(q,self.router)
        self.assertEqual([f.paper_id for f in facets],['a','b'])
        self.assertNotIn('RocketQA',facets[0].question)
        self.assertIn('训练损失',facets[0].question)
        self.assertIn('原公式',facets[0].question)
        self.assertEqual(paper_facets(q,self.router,allowed_ids=['a']),[])
        for q in ['有没有一种训练损失？','DPR 的训练损失？','ABC 与 DPR 的训练损失？']:
            self.assertEqual(paper_facets(q,self.router),[])

    def test_ambiguous_alias_never_creates_two_fictional_facets(self):
        r=PaperRouterV2(self.catalog,['a','b'],{'a':['ABC'],'b':['ABC']})
        self.assertEqual(paper_facets('ABC 怎么训练？',r),[])

    def test_contrastive_topic_does_not_force_global_but_real_comparison_does(self):
        self.assertEqual(self.router.route('RocketQA 的对比学习损失是什么？')['paper_ids'],['b'])
        self.assertEqual(self.router.route('对比 DPR 与 RocketQA 的对比学习损失')['mode'],'global')

    def test_candidate_reservation_keeps_fixed_limit_and_source_bodies(self):
        original=[hit(str(i)) for i in range(12)]
        facets=[[hit('a1')],[hit('b1','b')]]
        before=copy.deepcopy([original,facets])
        chosen=balanced_candidates(original,facets,original+sum(facets,[]),6)
        self.assertEqual(len(chosen),6)
        self.assertTrue({'a1','b1'}.issubset(h['chunk_id'] for h in chosen))
        self.assertEqual([original,facets],before)
        self.assertEqual(len({h['chunk_id'] for h in chosen}),6)


class V2LexicalTests(unittest.TestCase):
    def test_chinese_tokens_and_real_english_bridge_without_gold_translation(self):
        self.assertIn('检索',lexical_tokens('检索训练'))
        self.assertNotIn('索',lexical_tokens('检索训练'))
        chunks=[Chunk(paper_id='a',document_sha256='a'*64,chunker_version='fixture',chunk_index=0,
                      text='The contrastive loss trains a positive passage against negative passages.'),
                Chunk(paper_id='b',document_sha256='b'*64,chunker_version='fixture',chunk_index=0,
                      text='Weather forecasts and cloud observations.'),
                Chunk(paper_id='c',document_sha256='c'*64,chunker_version='fixture',chunk_index=0,
                      text='Medical research about heart disease.')]
        index=LexicalIndexV2(chunks)
        self.assertEqual(index.search('如何计算对比损失？')[0]['paper_id'],'a')
        self.assertIn('contrastive',index.query('对比损失')['tokens'])
        self.assertEqual(index.search('对比损失',paper_ids=['b']),[])
        self.assertEqual(LexicalIndexV2(chunks,expand=False).search('如何计算对比损失？'),[])


class V2PipelineTests(unittest.TestCase):
    def pipeline(self):
        settings=RetrievalSettings(candidate_k=12,rerank_k=6,context_char_budget=16000)
        rows=[hit(f'{pid}{i}',pid,f'{pid} source body {i}') for pid in ['a','b'] for i in range(12)]
        calls=[]
        class Models:
            def query_vector(self,q): return q
            def rerank(self,q,hs):
                calls.append(('rerank',q,[h['chunk_id'] for h in hs]))
                return [{**h,'score':100-i} for i,h in enumerate(hs)]
        class Store:
            def validate_collection(self): pass
            def search(self,vector,snapshot,top_k,paper_ids,exact):
                calls.append(('search',vector,top_k,paper_ids))
                return [h for h in rows if not paper_ids or h['paper_id'] in paper_ids][:top_k]
            def close(self): pass
        class Lexical:
            def query(self,q): return {'text':q,'tokens':[],'expanded_terms':{}}
        p=PipelineV2(settings,models=Models(),store=Store())
        p._router=PaperRouterV2([{'paper_id':'a','title':'Alpha: Method A'},
                                {'paper_id':'b','title':'Beta: Method B'}],['a','b'],{'a':['Alpha'],'b':['Beta']})
        p.snapshot=lambda:({'snapshot_id':'snapshot','documents':{'a':'a'*64,'b':'b'*64}},Lexical())
        return p,calls

    def test_full_scoped_path_filters_before_retrieval_and_balances_candidates(self):
        p,calls=self.pipeline()
        r=p.retrieve('对比 Alpha 与 Beta 的损失。请展示公式。',mode='dense',collect_trace=True)
        searches=[c for c in calls if c[0]=='search']
        self.assertEqual([c[3] for c in searches],[None,['a'],['b']])
        self.assertEqual(sum(c[2] for c in searches),12)
        self.assertEqual(len(r['reranked_candidates']),6)
        self.assertEqual({h['paper_id'] for h in r['evidence'][:3]},{'a','b'})
        self.assertTrue(all(h['text'].startswith(h['paper_id']+' source body') for h in r['reranked_candidates']))
        self.assertFalse(r['query_plan']['api_called'])
        self.assertEqual(r['context_policy']['eligible'],'all_reranked_candidates')

    def test_decomposition_can_be_disabled_and_manual_scope_cannot_escape(self):
        p,calls=self.pipeline()
        p.policy=RetrievalPolicyV2(decomposition=False)
        p.retrieve('对比 Alpha 与 Beta 的损失',mode='dense')
        self.assertEqual(len([c for c in calls if c[0]=='search']),1)
        calls.clear()
        p.policy=RetrievalPolicyV2()
        r=p.retrieve('对比 Alpha 与 Beta 的损失',mode='dense',paper_ids=['a'])
        self.assertEqual({h['paper_id'] for h in r['reranked_candidates']},{'a'})
        self.assertEqual(r['query_plan']['facets'],[])

    def test_context_scans_beyond_top5_and_respects_serialized_budget(self):
        from rag.generation import select_context
        rows=[hit(str(i),text='x'*1000) for i in range(20)]
        # Different bodies are needed: exact duplicates are correctly suppressed.
        rows=[{**h,'text':h['text']+str(i)} for i,h in enumerate(rows)]
        selected,used=select_context(rows,16000)
        self.assertGreater(len(selected),5)
        self.assertLessEqual(used,16000)


class FormulaRankingV2Tests(unittest.TestCase):
    def fixture(self):
        formulas=[{'paper_id':'a','element_id':str(i),'document_sha256':'a'*64,'pdf_page':1,
                   'equation_number':str(i),'image_sha256':'unchanged'+str(i),
                   'text_proxy':'untrusted '+str(i),'matches':[{'method':'adjacent_explanation','source_chunk_id':'c'}]}
                  for i in range(15)]
        elements=[SimpleNamespace(element_id=str(i),paper_id='a',title='A',section_title='loss',context=[],raw_text=str(i))
                  for i in range(15)]
        class Store:
            def document(self,*args): return {'elements':elements}
        class Models:
            def rerank(self,q,hs):
                return sorted([{**h,'score':int(h['chunk_id'].split('/')[-1])} for h in hs],key=lambda h:-h['score'])
        return formulas,Store(),Models()

    def test_rank_before_limit_preserves_images_and_untrusted_proxy(self):
        fs,store,models=self.fixture();before=copy.deepcopy(fs)
        attached={'formulas':fs,'evidence':[{'paper_id':'a','chunk_id':'c','formula_ids':[str(i) for i in range(15)]}],
                  'formula_retrieval':{'diagnostics':[],'matched_before_limit':15}}
        with patch('rag.formula_retrieval_v2.attach_formulas',return_value=attached):
            r=attach_formulas_v2('说明损失',[],paper_ids=['a'],store=store,models=models,limit=2)
        self.assertEqual([f['element_id'] for f in r['formulas']],['14','13'])
        self.assertEqual(fs,before)
        self.assertEqual(r['formulas'][0]['image_sha256'],'unchanged14')
        self.assertEqual(r['formulas'][0]['text_proxy'],'untrusted 14')
        self.assertEqual(r['evidence'][0]['formula_ids'],['13','14'])

    def test_exact_lookup_without_scope_or_missing_number_returns_no_unrelated_formulas(self):
        fs,store,models=self.fixture()
        for question,scope in [('公式(1)',None),('公式(9999)',['a'])]:
            attached={'formulas':copy.deepcopy(fs),'evidence':[],'formula_retrieval':{'diagnostics':[]}}
            with patch('rag.formula_retrieval_v2.attach_formulas',return_value=attached):
                r=attach_formulas_v2(question,[],paper_ids=scope,store=store,models=models)
            self.assertEqual(r['formulas'],[])

    def test_formula_rescue_follows_only_existing_source_links_and_prefers_body(self):
        candidates=[hit('early','a'),hit('body','a'),hit('other','b')]
        fs=[{'paper_id':'a','matches':[{'source_chunk_id':'early','method':'adjacent_explanation'},
                                      {'source_chunk_id':'body','method':'formula_inside_chunk'}]},
            {'paper_id':'b','matches':[{'source_chunk_id':'other','method':'formula_inside_chunk'}]}]
        attached={'formulas':fs,'formula_retrieval':{'matched_before_limit':2,'rerank_pairs':4}}
        with patch('rag.formula_retrieval_v2.attach_formulas_v2',return_value=attached):
            ids,info=rescue_formula_sources('对比两个方法的公式',candidates,
                [{'paper_id':'a','question':'A'},{'paper_id':'b','question':'B'}],models=object(),documents={})
        self.assertEqual(ids,['body','other'])
        self.assertEqual(info['rerank_pairs'],4)
        self.assertEqual(candidates[1]['text'],'Original body')

    def test_nonmath_guard_distinguishes_inspection_from_requesting_a_formula(self):
        fs,store,_=self.fixture()
        for question,expected in [('该方案标签是否被错误展示成数学公式？',0),('请展示数学公式，不是代码。',12)]:
            attached={'formulas':copy.deepcopy(fs),'evidence':[],'formula_retrieval':{'diagnostics':[]}}
            with patch('rag.formula_retrieval_v2.attach_formulas',return_value=attached):
                r=attach_formulas_v2(question,[],store=store,rank=False)
            self.assertEqual(len(r['formulas']),expected)


if __name__=='__main__': unittest.main()

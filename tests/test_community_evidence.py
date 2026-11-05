"""Scope, source joins, non-oracle ranking and compute-budget controls; no model/API."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rag.benchmark import score_evidence
from rag.community_evaluation import run_arm,run_case
from rag.community_retrieval import (CommunityOriginalRetriever,CommunityPolicy,OriginalSources,
    merge_candidates,novel_text,pack_context,token_pieces)

SHA='a'*64
PAGE='abcdefghijklmnopqrstuvwxyz'


def hit(identifier,a,b,branch='fulltext',pid='p'):
    return {'chunk_id':identifier,'paper_id':pid,'document_sha256':SHA,'pdf_page':1,'title':'Paper '+pid,
        'text':PAGE[a:b],'source_spans':[{'block_number':0,'source_start':a,'source_end':b,'text_start':0,'text_end':b-a}],
        'retrieval_paths':[{'branch':branch}], 'score':1.}


def gold(a=0,b=10):
    return {'question_id':'q','question':'A question','category':'single_paper','answerable':True,
        'gold_sources_exhaustive':True,'evidence_units':[{'unit_id':'u','alternatives':[[{
            'paper_id':'p','document_sha256':SHA,'pdf_page':1,'char_start':a,'char_end':b,'quote':PAGE[a:b]}]]}]}


class Tokenizer:
    def __call__(self,text,**kw):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}


class Ranker:
    def __init__(self):self.calls=[]
    def rerank(self,q,hits):
        self.calls.append((q,[h['text'] for h in hits]));return [{**h,'score':float(i)} for i,h in enumerate(reversed(hits))]


class SourceAndBudgetTests(unittest.TestCase):
    def test_source_identity_is_coordinates_not_chunk_id(self):
        resolver=OriginalSources.__new__(OriginalSources)
        resolver.manifest={'documents':{'p':SHA}}
        resolver.corpus=SimpleNamespace(papers={'p':{}},canonical_pages=lambda p:{1:{'text':PAGE,'sha256':hashlib.sha256(PAGE.encode()).hexdigest()}})
        a=resolver.canonical(hit('body',2,8));b=resolver.canonical(hit('anchor',2,8,'community_claim'))
        self.assertEqual(a['chunk_id'],b['chunk_id'])
        changed=hit('bad',2,8);changed['text']='changed'
        with self.assertRaises(ValueError):resolver.canonical(changed)
        changed=hit('bad',2,8);changed['document_sha256']='b'*64
        with self.assertRaises(ValueError):resolver.canonical(changed)

    def test_exact_duplicate_paths_are_merged_before_candidate_cutoff(self):
        body=hit('same',0,8);anchor=hit('same',0,8,'community_claim')
        merged,trace=merge_candidates([body],[anchor])
        self.assertEqual(len(merged),1)
        self.assertEqual({p['branch'] for p in merged[0]['retrieval_paths']},{'fulltext','community_claim'})
        self.assertEqual(trace['exact_duplicate_count'],1)

    def test_containment_dedup_does_not_manufacture_larger_passages(self):
        a,b=hit('long',0,10),hit('short',2,5,'dimension_claim')
        merged,trace=merge_candidates([a,b],[])
        self.assertEqual([h['text'] for h in merged],[PAGE[:10]])
        self.assertEqual(trace['contained_duplicate_count'],1)
        self.assertEqual(len(merged[0]['retrieval_paths']),2)
        merged,_=merge_candidates([hit('a',0,8)],[hit('b',5,12,'community_claim')])
        self.assertEqual({h['text'] for h in merged},{PAGE[:8],PAGE[5:12]})

    def test_overlap_subtraction_preserves_source_coverage_and_budget(self):
        packed,usage=pack_context([hit('a',0,8),hit('b',5,13)],1000)
        spans=[(s['source_start'],s['source_end']) for h in packed for s in h['source_spans']]
        self.assertEqual(spans,[(0,8),(8,13)])
        self.assertEqual(usage['overlap_characters_removed'],3)
        self.assertTrue(score_evidence(gold(0,13),packed,{('p',SHA,1):PAGE},k=2)['all_required_evidence'])
        for h in packed:
            for s in h['source_spans']:self.assertEqual(h['text'][s['text_start']:s['text_end']],PAGE[s['source_start']:s['source_end']])
        serialized=[{'id':h['citation_id'],'paper_id':h['paper_id'],'title':h['title'],'pdf_page':1,'text':h['text']} for h in packed]
        self.assertEqual(usage['used_chars'],len(json.dumps(serialized,ensure_ascii=False)))
        few,tiny=pack_context([hit('a',0,8),hit('b',5,13)],usage['used_chars']-1)
        self.assertEqual(len(few),1);self.assertLessEqual(tiny['used_chars'],usage['used_chars']-1)

    def test_disjoint_residual_fragments_keep_exact_offsets(self):
        result=novel_text(hit('a',0,12),[(3,7)])
        self.assertEqual(result['text'],PAGE[:3]+'\n\n'+PAGE[7:12])
        self.assertEqual([(s['source_start'],s['source_end']) for s in result['source_spans']],[(0,3),(7,12)])

    def test_token_pieces_are_exact_substrings_without_silent_truncation(self):
        text=' abcdef ghijk '
        pieces=list(token_pieces(text,Tokenizer(),5,1))
        self.assertGreater(len(pieces),1)
        for a,b,p,n in pieces:self.assertEqual(p,text[a:b]);self.assertLessEqual(n,5)

    def test_all_arms_share_original_question_and_rerank_ceiling(self):
        policy=CommunityPolicy(rerank_limit=2,context_chars=1000);models=Ranker()
        base=[hit('a',0,3),hit('b',5,8),hit('c',10,14)]
        for additions in [[],[hit('d',17,20,'dimension_claim')],[hit('e',20,23,'community_claim')]]:
            result=run_arm('same question',base,additions,models,policy)
            self.assertEqual(result['rerank_pairs'],2)
            self.assertEqual(len(result['reranked_candidates']),2)
        self.assertTrue(all(q=='same question' for q,_ in models.calls))

    def test_union_reranking_can_regress_even_if_baseline_is_present(self):
        models=Ranker();policy=CommunityPolicy(rerank_limit=3,context_chars=102)
        base=[hit('gold',0,10)]
        old=run_arm('same',base,[],models,policy)
        new=run_arm('same',base,[hit('zbad',15,25,'community_claim')],models,policy)
        pages={('p',SHA,1):PAGE}
        self.assertTrue(score_evidence(gold(),old['rankings']['context_budget'],pages,k=1)['all_required_evidence'])
        self.assertFalse(score_evidence(gold(),new['rankings']['context_budget'],pages,k=1)['all_required_evidence'])


class BranchSelectionTests(unittest.TestCase):
    def setUp(self):
        ids=['a','b','c','d'];vectors=np.asarray([[1,0],[.99,.1],[.95,.31],[0,1]],dtype=np.float32)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        claims={p:{'claim_id':p,'paper_id':p,'dimension':'method','text':p,'source_ids':[p]} for p in ids}
        def source(p):return {'paper_id':p,'document_sha256':SHA,'pdf_page':1,'text':'Original '+p,
            'source_start':0,'source_end':10,'title':'Paper '+p,'bbox':[0,0,10,10]}
        self.corpus=SimpleNamespace(build_id='graph',papers={p:{} for p in ids},claims=claims,
            vectors={'method':(ids,vectors)},communities={'method':[
                {'community_id':'g1','members':['a','d']},{'community_id':'g2','members':['b','c']}]},
            memberships={'method':{'a':'g1','b':'g2','c':'g2','d':'g1'}},source=source)
        models=SimpleNamespace(tokenizer=Tokenizer(),settings=SimpleNamespace(chunk_size=384,chunk_overlap=0))
        self.retriever=CommunityOriginalRetriever(self.corpus,models,ids,vectors,CommunityPolicy(dimension='method',paper_limit=1,community_limit=1))
        self.manifest={'documents':{p:SHA for p in [*ids,'outside']}}

    def test_membership_can_change_selection_beyond_direct_vectors(self):
        a,bt=self.retriever.retrieve('methods','B',{'mode':'global'},None,self.manifest,np.array([1,0]))
        b,ct=self.retriever.retrieve('methods','C',{'mode':'global'},None,self.manifest,np.array([1,0]))
        self.assertEqual(bt['selected_paper_ids'],['a'])
        self.assertEqual(ct['selected_paper_ids'],['b'])
        self.assertEqual(ct['selected_community_ids'],['g2'])
        self.assertNotIn('outside',bt['scope_paper_ids'])
        self.assertEqual(bt['claim_rerank_pairs'],0)

    def test_single_paper_branch_is_skipped_without_source_reads(self):
        self.corpus.source=lambda p:(_ for _ in ()).throw(AssertionError('should not read anchors'))
        for variant in ['B','C']:
            hits,trace=self.retriever.retrieve('question',variant,{'mode':'single_paper','paper_ids':['a']},None,self.manifest,np.array([1,0]))
            self.assertEqual(hits,[]);self.assertEqual(trace['status'],'protected_single_paper')

    def test_explicit_comparison_never_uses_other_graph_papers(self):
        with patch('rag.community_retrieval.paper_facets',return_value=[SimpleNamespace(paper_id='c'),SimpleNamespace(paper_id='d')]):
            for variant in ['B','C']:
                hits,trace=self.retriever.retrieve('comparison',variant,{'mode':'global'},object(),self.manifest,np.array([1,0]))
                self.assertEqual(set(trace['scope_paper_ids']),{'c','d'})
                self.assertTrue(all(h['paper_id'] in {'c','d'} for h in hits))

    def test_wrong_source_version_fails_instead_of_falling_back(self):
        self.manifest['documents']['a']='b'*64
        with self.assertRaisesRegex(ValueError,'scope'):
            self.retriever.retrieve('methods','B',{'mode':'global'},None,self.manifest,np.array([1,0]))


if __name__=='__main__':unittest.main()

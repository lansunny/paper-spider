"""Evaluation denominators, blinding and source-quote constraints, with no API calls."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from rag.reading_evaluation import list_metrics,navigation_payload,validate_relevance,validate_audit,direct_candidates,normalized
from rag.paper_graph.schema import DIMENSIONS


class ReadingEvaluationTests(unittest.TestCase):
    def test_fixed_slots_do_not_reward_returning_fewer_papers(self):
        value=list_metrics(['a','b'],{'a':3,'b':2,'c':1})
        self.assertEqual(value['useful_slots_at5'],.4)
        self.assertEqual(value['precision_returned'],1)
        self.assertEqual(value['returned'],2)

    def test_pdf_word_hyphenation_is_typography_but_numeric_signs_are_not(self):
        self.assertEqual(normalized('a rule-guided evalua- tion task'),normalized('a rule-guided evaluation task'))
        self.assertNotEqual(normalized('score -0.5'),normalized('score 0.5'))

    def test_failure_and_unknown_labels_do_not_become_perfect_precision(self):
        empty=list_metrics([],{'a':3})
        self.assertEqual(empty['useful_slots_at5'],0)
        self.assertIsNone(empty['precision_returned'])
        unknown=list_metrics(['a'],{'a':None})
        self.assertEqual(unknown['useful_count'],0)
        self.assertEqual(unknown['unknown_count'],1)
        self.assertIsNone(unknown['ndcg_at5_pooled'])

    def payload(self):
        return {'seed':{'sources':[{'id':'SEED1','text':'A paper evaluates retrieval and answer faithfulness.'}]},
            'candidates':[{'id':'P01','sources':[{'id':'P01S1','text':'We propose an evaluator trained with synthetic examples.'}]}]}

    def test_useful_grades_require_seed_and_candidate_original_quotes(self):
        value={'ratings':[{'paper':'P01','grade':3,'reason':'Direct comparison','role':'Evaluation methods',
            'quotes':[{'snippet_id':'SEED1','quote':'retrieval and answer faithfulness'},
                      {'snippet_id':'P01S1','quote':'an evaluator trained with synthetic examples'}]}]}
        validate_relevance(value,self.payload())
        value['ratings'][0]['quotes']=value['ratings'][0]['quotes'][:1]
        self.assertIsNone(validate_relevance(value,self.payload())['ratings'][0]['grade'])

    def test_judge_cannot_use_invented_quotes_or_other_papers(self):
        value={'ratings':[{'paper':'P01','grade':0,'reason':'Not relevant','role':'Other topic',
            'quotes':[{'snippet_id':'P01S1','quote':'This invented sentence does not occur.'}]}]}
        parsed=validate_relevance(value,self.payload())
        self.assertIsNone(parsed['ratings'][0]['grade'])
        self.assertTrue(parsed['ratings'][0]['quote_errors'])

    def test_supplied_original_anchor_ids_resolve_to_the_same_visible_excerpt(self):
        payload=self.payload();payload['candidates'][0]['sources'][0]['anchor_id']='original-anchor'
        value={'ratings':[{'paper':'P01','grade':0,'reason':'Not relevant','role':'Other topic',
            'quotes':[{'snippet_id':'original-anchor','quote':'an evaluator trained with synthetic examples'}]}]}
        parsed=validate_relevance(value,payload)
        self.assertEqual(parsed['ratings'][0]['quotes'][0]['snippet_id'],'P01S1')

    def test_audit_only_accepts_the_sources_actually_cited(self):
        payload={'items':[{'item_id':'i','citations':['S1']}],
            'sources':[{'id':'S1','text':'Reference-free evaluation still uses retrieved context.'},
                       {'id':'S2','text':'Other evidence cannot rescue the wrong citation.'}]}
        value={'ratings':[{'item_id':'i','verdict':'partial','reason':'One unsupported fact','unsupported_parts':['no context'],
            'quotes':[{'snippet_id':'S2','quote':'Other evidence cannot rescue'}]}]}
        self.assertEqual(validate_audit(value,payload)['ratings'][0]['verdict'],'unclear')
        value['ratings'][0]['quotes']=[{'snippet_id':'S1','quote':'still uses retrieved context'}]
        validate_audit(value,payload)
        value['ratings'][0]['verdict']='full'
        self.assertEqual(validate_audit(value,payload)['ratings'][0]['verdict'],'unclear')

    def test_navigation_ablation_preserves_candidates_and_removes_raw_graph_metadata(self):
        corpus=SimpleNamespace(build_id='frozen',papers={'seed':{'title':'Seed'},'a':{'title':'Candidate'}},
            profiles={(p,d):{'description':p+' '+d} for p in ('seed','a') for d in DIMENSIONS})
        pool=[{'paper_id':'a','score':.99,'directions':['graph'],'communities':{'topic':'t'}}]
        cards=[{'title':'Topic','paper_ids':['seed','a'],'scope':[{'text':'A sourced clue'}],
            'routes':[{'name':'Method','paper_ids':['a'],'overview':{'text':'A candidate approach'}}]}]
        b=navigation_payload(corpus,{'paper_id':'seed','goal':'compare'},pool,[])
        c=navigation_payload(corpus,{'paper_id':'seed','goal':'compare'},pool,cards)
        self.assertEqual(b['candidates'],c['candidates'])
        self.assertEqual(set(b['candidates'][0]),{'paper_id','title','descriptions'})
        self.assertEqual(b['community_navigation'],[])
        self.assertTrue(c['community_navigation'])

    def test_direct_vector_baseline_never_recommends_the_seed(self):
        values=np.array([[1.,0.],[.8,.6],[0.,1.]],dtype=np.float32)
        corpus=SimpleNamespace(vectors={d:(['seed','a','b'],values) for d in DIMENSIONS})
        result=direct_candidates(corpus,'seed',np.array([1.,0.]),12)
        self.assertEqual([v['paper_id'] for v in result],['a','b'])


if __name__=='__main__':unittest.main()

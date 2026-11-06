"""Regression tests for attribution, precise reading entrances and bounded review cost."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from rag.generation import GenerationSettings
from rag.paper_graph.extraction import request_json
from rag.reading_grounding import (apply_review, exact_quote, ground_advice, normalized_offsets, numbers,
                                  review_payload, validate_review)


def source(pid,sid,text):
    return {'paper_id':pid,'citation_id':sid,'title':pid,'pdf_page':3,'document_sha256':'a'*64,'text':text,
            'study_url':'/study?paper='+pid+'&page=3',
            'source_spans':[{'text_start':0,'text_end':len(text),'source_start':100,'source_end':100+len(text)}]}


class ReadingGroundingTests(unittest.TestCase):
    def setUp(self):
        self.evidence=[source('seed','S1','Our score uses question embeddings and cosine similarity.'),
                       source('target','S2','The sampled human annotation baseline uses 1,350 annotations. ARES uses 300 validation annotations.')]
        self.seed={'paper_id':'seed','title':'Seed'};self.candidates=[{'paper_id':'target','title':'ARES'}]
        self.answer={'recommendations':[{'paper_id':'target','status':'supported',
            'why_read':[{'text':'Compare embedding similarity with a calibrated evaluator.','citations':['S1','S2']}],
            'differences':[{'text':'Compare their annotation requirements.','citations':['S1','S2']}],
            'focus':[{'text':'Read the validation annotation comparison.','citations':['S2']}],
            'evidence_gaps':[]}],'reading_order':['target']}

    def payload(self):return review_payload(self.answer,self.evidence,self.seed,self.candidates,'Compare evaluators')

    def reviews(self):
        values=[]
        for item in self.payload()['items']:
            values.append({'item_id':item['item_id'],'verdict':'full','reason':'Supported by cited source.',
                'location_type':'direct' if item['kind']=='focus' else 'not_applicable',
                'quotes':[{'snippet_id':sid,'quote':self.evidence[0 if sid=='S1' else 1]['text']} for sid in item['citations']]})
        return {'ratings':values}

    def test_quotes_keep_exact_original_offsets_through_ligatures_and_line_wraps(self):
        text='🕷 Reading: efﬁcient evalua-\n tion uses score -0.5.'
        hit=source('p','S1',text)
        quote=exact_quote(hit,'efficient evaluation uses score -0.5.')
        self.assertEqual(text[quote['text_start']:quote['text_end']],quote['quote'])
        self.assertEqual(quote['source_spans'][0]['source_start'],100+text.index('efﬁcient'))
        self.assertEqual(quote['source_spans'][0]['source_end'],100+len(text))
        with self.assertRaises(ValueError):exact_quote(hit,'efficient evaluation uses score 0.5.')
        self.assertNotEqual(normalized_offsets('-0.5')[0],normalized_offsets('0.5')[0])

    def test_quotes_crossing_source_spans_preserve_each_coordinate(self):
        hit=source('p','S1','first statement\n\nsecond statement')
        hit['source_spans']=[{'text_start':0,'text_end':15,'source_start':10,'source_end':25},
                             {'text_start':17,'text_end':33,'source_start':100,'source_end':116}]
        quote=exact_quote(hit,hit['text'])
        self.assertEqual(quote['source_spans'],[{'source_start':10,'source_end':25},{'source_start':100,'source_end':116}])

    def test_full_review_must_support_both_papers_and_quote_real_cited_text(self):
        bad=self.reviews();bad['ratings'][0]['quotes']=bad['ratings'][0]['quotes'][:1]
        value=validate_review(bad,self.payload(),self.evidence)
        self.assertEqual(value['ratings'][0]['verdict'],'unclear')
        bad=self.reviews();bad['ratings'][0]['quotes'][0]['quote']='A completely invented quote for the evaluator.'
        self.assertEqual(validate_review(bad,self.payload(),self.evidence)['ratings'][0]['verdict'],'unclear')

    def test_model_full_cannot_hide_a_numeric_mismatch(self):
        self.answer['recommendations'][0]['focus'][0]['text']='ARES uses 1350 validation annotations.'
        bad=self.reviews();bad['ratings'][2]['quotes']=[{'snippet_id':'S2','quote':'ARES uses 300 validation annotations.'}]
        checked=validate_review(bad,self.payload(),self.evidence)
        self.assertEqual(checked['ratings'][2]['verdict'],'unclear')
        self.assertTrue(any('Numeric' in s for s in checked['ratings'][2]['validation_errors']))

    def test_matching_numbers_do_not_certify_experimental_subjects(self):
        self.answer['recommendations'][0]['why_read'][0]['text']='ARES requires 1,350 annotations.'
        checked=validate_review(self.reviews(),self.payload(),self.evidence)
        self.assertEqual(checked['ratings'][0]['verdict'],'unclear')
        self.assertEqual(numbers('GPT-3.5-turbo, DeBERTa-v3-Large and Qwen2.5'),set())
        self.assertEqual(numbers('结果为-0.5，人工标注1,350条，下降78%。'),{'-0.5','1350','78%'})

    def test_procedural_location_is_conservative_even_if_model_calls_it_full(self):
        self.answer['recommendations'][0]['focus'][0]['text']='重点看 PPI 修正流程与指标计算定义。'
        checked=validate_review(self.reviews(),self.payload(),self.evidence)
        self.assertEqual(checked['ratings'][2]['location_type'],'source_excerpt')
        clean=apply_review(self.answer,checked,self.evidence)['recommendations'][0]
        self.assertEqual(clean['focus'][0]['location_type'],'source_excerpt')
        self.assertTrue(clean['focus'][0]['support_quotes'])

    def test_wrong_numeric_subject_is_removed_even_when_number_occurs_in_source(self):
        self.answer['recommendations'][0]['why_read'][0]['text']='ARES itself requires 1,350 annotations.'
        raw=self.reviews();raw['ratings'][0].update(verdict='unsupported',reason='1,350 belongs to the baseline, not ARES.')
        reviewed=validate_review(raw,self.payload(),self.evidence)
        clean=apply_review(self.answer,reviewed,self.evidence)['recommendations'][0]
        self.assertEqual(clean['status'],'insufficient_evidence')
        self.assertEqual(clean['why_read'],[])
        self.assertEqual(clean['differences'],[])
        self.assertNotIn('ARES itself requires 1,350 annotations.',
                         ' '.join(i['text'] for field in ('why_read','differences','focus') for i in clean[field]))

    def test_result_page_is_not_published_as_an_algorithm_location(self):
        raw=self.reviews();raw['ratings'][2].update(verdict='partial',location_type='overview',reason='Results only; no algorithm steps.')
        clean=apply_review(self.answer,validate_review(raw,self.payload(),self.evidence),self.evidence)['recommendations'][0]
        self.assertEqual(clean['focus'][0]['location_type'],'source_excerpt')
        self.assertNotIn('validation annotation comparison',clean['focus'][0]['text'])
        raw=self.reviews();raw['ratings'][2]['location_type']='overview'
        clean=apply_review(self.answer,validate_review(raw,self.payload(),self.evidence),self.evidence)['recommendations'][0]
        self.assertEqual(clean['focus'][0]['location_type'],'overview')
        self.assertTrue(clean['focus'][0]['support_quotes'])

    def test_failed_review_never_exposes_original_unverified_comparisons(self):
        with tempfile.TemporaryDirectory() as temp, patch('rag.reading_grounding.request_step',side_effect=RuntimeError('offline')) as step:
            answer,review,record=ground_advice(Path(temp),self.answer,self.evidence,self.seed,self.candidates,'Compare',GenerationSettings())
        self.assertEqual(step.call_args.kwargs['max_attempts'],1)
        self.assertEqual(review['status'],'unavailable')
        self.assertEqual(answer['recommendations'][0]['status'],'insufficient_evidence')
        self.assertEqual(answer['recommendations'][0]['why_read'],[])
        self.assertEqual(record['usage']['total_tokens'],0)

    def test_complete_review_retains_original_supported_claims_and_exact_quotes(self):
        review=validate_review(self.reviews(),self.payload(),self.evidence)
        clean=apply_review(self.answer,review,self.evidence)['recommendations'][0]
        self.assertEqual(clean['status'],'supported')
        self.assertEqual(clean['why_read'][0]['text'],self.answer['recommendations'][0]['why_read'][0]['text'])
        self.assertEqual(clean['focus'][0]['location_type'],'direct')
        self.assertTrue(clean['why_read'][0]['support_quotes'])

    def test_review_is_bounded_to_one_http_response_on_invalid_json(self):
        requests=[]
        def transport(request):
            requests.append(request)
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}],
                                            'usage':{'total_tokens':12}})
        settings=GenerationSettings(base_url='https://example.test/v1',api_key='test')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                request_json('test',{},settings,Path(temp),lambda v:v,
                             transport=httpx.MockTransport(transport),max_attempts=1)
            failures=list(Path(temp).glob('*.failed.json'))
            self.assertEqual(len(json.loads(failures[0].read_text())['attempts']),1)
        self.assertEqual(len(requests),1)


if __name__=='__main__':unittest.main()

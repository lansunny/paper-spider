"""Graph decision paths, pair citations, source budgets and job boundaries; no API calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from backend.related_reading import RelatedReadingJobs, create_related_reading_router
from rag.related_reading import (ReadingRequest, RelatedReadingEngine, balanced_context, fallback_plan,
    graph_candidates, validate_advice, validate_plan)
from rag.paper_graph.build import prepare, reuse_published_profiles
from rag.paper_graph.schema import SOURCE_VERSION
from rag.paper_graph.extraction import PROMPT_VERSION
from rag.reviews.corpus import ReviewCorpus
from rag.storage import atomic_json
from test_paper_graph import PaperGraphTests


def evidence(pid,identifier,start=0,end=10):
    return {'chunk_id':identifier,'citation_id':identifier,'paper_id':pid,'document_sha256':'a'*64,
        'pdf_page':1,'title':pid,'text':'abcdefghij'[start:end],
        'source_spans':[{'source_start':start,'source_end':end,'text_start':0,'text_end':end-start,'block_number':0}]}


class RelatedReadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=PaperGraphTests();cls.fixture.setUp();cls.fixture.build()
        cls.root=cls.fixture.root;cls.corpus=ReviewCorpus(cls.root)

    @classmethod
    def tearDownClass(cls):cls.fixture.tearDown()

    def test_graph_connections_actually_determine_candidates(self):
        candidates=graph_candidates(self.corpus,'p0','compare methods',limit=5)
        self.assertEqual({c['paper_id'] for c in candidates},{'p1','p2'})
        self.assertTrue(all(c['edge_weights'] for c in candidates))
        with self.assertRaisesRegex(ValueError,'尚未纳入'):
            graph_candidates(self.corpus,'outside','compare')
        views=json.loads((self.corpus.directory/'views.json').read_text('utf-8'))
        for view in views.values():view['edges']=[]
        with patch.object(self.corpus,'assert_current'), patch('rag.related_reading.json.loads',return_value=views):
            self.assertEqual(graph_candidates(self.corpus,'p0','compare'),[])

    def test_followups_cannot_add_papers_or_skip_candidates(self):
        candidates=graph_candidates(self.corpus,'p0','compare')
        plan=fallback_plan(candidates,'Seed paper','Compare evaluation mechanisms')
        validate_plan(plan,candidates)
        bad=deepcopy(plan);bad['questions'][0]['paper_ids']=['outside']
        with self.assertRaises(ValueError):validate_plan(bad,candidates)
        bad=deepcopy(plan)
        for q in bad['questions']:q['paper_ids']=['p1']
        with self.assertRaisesRegex(ValueError,'cover all'):validate_plan(bad,candidates)

    def test_empty_plan_is_valid_but_partially_empty_plans_are_not(self):
        candidates=graph_candidates(self.corpus,'p0','compare')
        empty={'selected_paper_ids':[],'questions':[]}
        self.assertEqual(validate_plan(empty,candidates),empty)
        self.assertEqual(fallback_plan([],'Seed','An unrelated goal'),empty)
        for bad in ({'selected_paper_ids':['p1'],'questions':[]},
                    {'selected_paper_ids':[],'questions':[{'question':'Compare methods','dimension':'method','paper_ids':['p1']}]}):
            with self.assertRaises(ValueError):validate_plan(bad,candidates)

    def test_no_recommendation_stops_before_retrieval_and_claim_embeddings(self):
        from rag.generation import GenerationSettings
        vector=np.zeros(1024,dtype=np.float32);vector[0]=1
        models=SimpleNamespace(query_vector=lambda q:vector)
        pipeline=SimpleNamespace(models=models)
        def plan(system,payload,settings,root,validator,purpose):
            value=validator({'selected_paper_ids':[],'questions':[]})
            return value,{'cache_hit':False,'api_attempts':1,'usage':{'total_tokens':30},'usage_current_request':{'total_tokens':30}}
        with (patch('rag.related_reading.generation_settings',return_value=GenerationSettings(api_key='test-key')),
             patch('rag.related_reading.request_step',side_effect=plan) as call,
             patch('rag.related_reading.current_manifest',side_effect=AssertionError('No retrieval needed')),
             patch('rag.related_reading.load_claim_vectors',side_effect=AssertionError('No embeddings needed'))):
            result=RelatedReadingEngine(self.root,self.corpus,pipeline).run({'paper_id':'p0','goal':'unrelated goal'})
        self.assertEqual(call.call_count,1)
        self.assertEqual(result['result_status'],'no_recommendations')
        self.assertEqual(result['evidence'],[])
        self.assertEqual(result['online_model_calls'],1)
        self.assertEqual(result['usage']['total_tokens'],30)

    def test_empty_recommendation_job_completes_and_can_be_read_again(self):
        jobs=RelatedReadingJobs(self.root,lambda:None)
        empty={'graph_build_id':self.corpus.build_id,'result_status':'no_recommendations',
               'evidence':[],'recommendations':[],'questions':[],'message':'没有直接符合目标的工作。'}
        with patch('backend.related_reading.RelatedReadingEngine') as engine:
            engine.return_value.run.return_value=empty
            job=jobs.start(ReadingRequest(paper_id='p0'))
            jobs.executor.shutdown(wait=True)
        completed=jobs.read(job['run_id'])
        self.assertEqual(completed['status'],'completed')
        self.assertEqual(completed['result']['recommendations'],[])

    def advice(self):
        return {'recommendations':[{'paper_id':'p1','status':'supported',
            'why_read':[{'text':'Compare two source-supported methods.','citations':['S1','S2']}],
            'differences':[{'text':'The methods use different evidence.','citations':['S1','S2']}],
            'focus':[{'text':'Read the method in the target paper.','citations':['S2']}],'evidence_gaps':[]}],
            'reading_order':['p1']}

    def test_recommendation_requires_both_original_papers(self):
        sources=[evidence('p0','S1'),evidence('p1','S2')]
        value=self.advice();validate_advice(value,sources,'p0',[{'paper_id':'p1'}])
        value['recommendations'][0]['why_read'][0]['citations']=['S2']
        with self.assertRaisesRegex(ValueError,'pair evidence'):
            validate_advice(value,sources,'p0',[{'paper_id':'p1'}])

    def test_long_paper_claim_can_keep_many_valid_sources_without_accepting_unknown_ids(self):
        from rag.paper_graph.schema import validate_description
        from test_paper_graph import description
        ids=['S'+str(i) for i in range(12)]
        value=description(ids[0])
        value['topic']['claims'][0]['source_ids']=ids
        validate_description(value,set(ids))
        value['topic']['claims'][0]['source_ids']=[*ids,'unknown']
        with self.assertRaises(ValueError):validate_description(value,set(ids))

    def test_large_cards_are_generated_with_separate_route_claim_scopes(self):
        from rag.generation import GenerationSettings
        from rag.reviews.cards import card_input,partitioned_card
        corpus=self.corpus;payload,aliases=card_input(corpus,corpus.communities['topic'][0])
        payload['route_cells']=[{'method_community_id':'m1','paper_ids':['p0']},
                                {'method_community_id':'m2','paper_ids':['p1','p2']}]
        calls=[]
        def step(system,data,settings,root,validator,purpose):
            calls.append(purpose)
            if purpose=='community_route_card':
                self.assertEqual(len(data['route_cells']),1)
                cell=data['route_cells'][0];members=cell['paper_ids']
                self.assertTrue(all(c['paper_id'] in members for c in data['claims']))
                ids=[next(c['id'] for c in data['claims'] if c['paper_id']==pid) for pid in members[:2]]
                statement={'text':'A scoped source-supported clue','support':'multi_paper' if len(ids)>1 else 'paper_specific','claim_ids':ids}
                value={'title':'Route topic','scope':[statement],'routes':[{'method_community_id':cell['method_community_id'],
                    'name':'Route','overview':statement,'differences':[],'experiments':[],'limitations':[],
                    'representative_paper_ids':members[:2]}]}
            else:
                ids=[data['routes'][0]['overview']['claim_ids'][0],data['routes'][1]['overview']['claim_ids'][0]]
                value={'title':'Combined topic','scope':[{'text':'Two supported directions','support':'multi_paper','claim_ids':ids}]}
            return validator(value),{'purpose':purpose,'usage':{'total_tokens':0}}
        with patch('rag.reviews.cards.request_step',side_effect=step):
            card,steps=partitioned_card(corpus,payload,aliases,GenerationSettings(),self.root)
        self.assertEqual(len(card['routes']),2)
        self.assertEqual(calls,['community_route_card','community_route_card','community_scope_card'])

    def test_differences_and_reading_locations_cannot_cite_the_wrong_paper(self):
        sources=[evidence('p0','S1'),evidence('p1','S2'),evidence('p2','S3')]
        for field,citations in [('differences',['S2']),('focus',['S1']),('why_read',['S1','S3'])]:
            value=self.advice();value['recommendations'][0][field][0]['citations']=citations
            with self.assertRaises(ValueError):validate_advice(value,sources,'p0',[{'paper_id':'p1'}])

    def test_unknown_citations_and_duplicate_reading_order_are_rejected(self):
        sources=[evidence('p0','S1'),evidence('p1','S2')]
        value=self.advice();value['recommendations'][0]['focus'][0]['citations']=['S99']
        with self.assertRaises(ValueError):validate_advice(value,sources,'p0',[{'paper_id':'p1'}])
        value=self.advice();value['reading_order']=['p1','p1']
        with self.assertRaises(ValueError):validate_advice(value,sources,'p0',[{'paper_id':'p1'}])

    def test_evidence_gaps_cannot_be_reported_as_supported_comparisons(self):
        value=self.advice();value['recommendations'][0]['status']='insufficient_evidence'
        with self.assertRaisesRegex(ValueError,'must not produce'):
            validate_advice(value,[evidence('p0','S1'),evidence('p1','S2')],'p0',[{'paper_id':'p1'}])

    def test_context_is_balanced_deduplicated_and_within_exact_budget(self):
        hits=[evidence('p0',f'a{i}') for i in range(8)]+[evidence('p1','b'),evidence('p2','c')]
        selected,usage=balanced_context(hits,['p0','p1','p2'],1500)
        self.assertEqual({h['paper_id'] for h in selected},{'p0','p1','p2'})
        self.assertLessEqual(usage['used_chars'],1500)
        self.assertEqual(sum(h['paper_id']=='p0' for h in selected),1)

    def test_jobs_reject_unknown_seed_and_path_traversal(self):
        jobs=RelatedReadingJobs(self.root,lambda:None)
        with self.assertRaises(KeyError):jobs.read('../outside')
        with self.assertRaisesRegex(ValueError,'尚未纳入'):
            jobs.start(ReadingRequest(paper_id='outside'))
        jobs.executor.shutdown(wait=True)

    def test_api_works_without_loading_models_and_does_not_generate_on_get(self):
        def forbidden():raise AssertionError('GET must not load inference models')
        app=FastAPI();app.include_router(create_related_reading_router(self.root,forbidden))
        with TestClient(app) as client:
            self.assertTrue(client.get('/api/related-reading/papers/p0').json()['available'])
            self.assertFalse(client.get('/api/related-reading/papers/outside').json()['available'])
            self.assertEqual(client.get('/api/related-reading/jobs/not-a-uuid').status_code,404)
            self.assertEqual(client.post('/api/related-reading',json={'paper_id':'outside'}).status_code,409)

    def test_evidence_only_runs_open_body_and_anchored_paths_without_generation(self):
        from rag.config import RetrievalSettings
        cfg=RetrievalSettings()
        vector=np.zeros(1024,dtype=np.float32);vector[0]=1
        models=SimpleNamespace(settings=cfg,query_vector=lambda q:vector,
            tokenizer=lambda text,**kw:{'offset_mapping':[(i,i+1) for i in range(len(text))]},
            rerank=lambda q,hits:[{**h,'score':1.} for h in hits])
        hits=[]
        for pid in self.corpus.papers:
            source=next(s for s in self.corpus.sources.values() if s['paper_id']==pid)
            hits.append({**source,'chunk_id':'body-'+pid,'title':pid,'score':1.})
        calls=[]
        def retrieve(question,**kwargs):
            calls.append(kwargs)
            return {'rankings':{'rrf20':hits,'union':hits}}
        pipeline=SimpleNamespace(settings=cfg,models=models,retrieve=retrieve)
        ids=sorted(self.corpus.claims);vectors=np.repeat(vector[None,:],len(ids),axis=0)
        manifest={'documents':{p:v['document_sha256'] for p,v in self.corpus.papers.items()}}
        with patch('rag.related_reading.current_manifest',return_value=manifest), \
             patch('rag.related_reading.load_claim_vectors',return_value=(ids,vectors,{'cache_hit':True})), \
             patch('rag.related_reading.request_step',side_effect=AssertionError('No generation in evidence-only mode')):
            result=RelatedReadingEngine(self.root,self.corpus,pipeline).run(ReadingRequest(paper_id='p0',generate=False))
        self.assertEqual(result['online_model_calls'],0)
        self.assertTrue(result['evidence'])
        self.assertTrue(all(c['auto_route'] is False and 'paper_ids' not in c for c in calls))
        self.assertLessEqual(result['context_usage']['used_chars'],16000)
        self.assertTrue(all(t['anchor_candidate_count']>0 for t in result['retrieval_trace']))

    def test_extension_reuses_profiles_with_original_model_provenance(self):
        from rag.paper_graph.build import profile_cache
        records=prepare(self.root,self.fixture.documents,self.fixture.config,self.fixture.generation)
        # Real published profiles include these versions; add them to the synthetic fixture bundle.
        path=self.corpus.directory/'profiles.jsonl'
        originals=path.read_bytes();manifest_path=self.corpus.directory/'manifest.json'
        manifest_bytes=manifest_path.read_bytes();pointer=self.root/'runtime/paper_graph/CURRENT.json';pointer_bytes=pointer.read_bytes()
        from rag.formula_store import digest_file
        try:
            values=[]
            for line in originals.decode().splitlines():
                p=json.loads(line);p['source_version']=SOURCE_VERSION;p['prompt_version']=PROMPT_VERSION;p['paper_id']=next(r['paper']['paper_id'] for r in records if r['cache_key']==p['cache_key']);values.append(p)
            path.write_text(''.join(json.dumps(v)+'\n' for v in values),encoding='utf-8')
            manifest=json.loads(manifest_bytes);manifest['artifacts']['profiles.jsonl']=digest_file(path);atomic_json(manifest_path,manifest)
            p=json.loads(pointer_bytes);p['manifest_sha256']=digest_file(manifest_path);atomic_json(pointer,p)
            new=prepare(self.root,self.fixture.documents,self.fixture.config,self.fixture.generation.model_copy(update={'model':'deepseek-flash'}))
            updated,reused=reuse_published_profiles(self.root,new)
            self.assertEqual(len(reused),3)
            self.assertEqual({r['model'] for r in reused},{'test-fixture'})
            self.assertEqual({r['cache_key'] for r in updated},{r['cache_key'] for r in records})
        finally:
            path.write_bytes(originals);manifest_path.write_bytes(manifest_bytes);pointer.write_bytes(pointer_bytes)


if __name__=='__main__':unittest.main()

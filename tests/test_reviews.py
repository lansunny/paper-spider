"""Review source scope, citation boundaries, packing and jobs; no paid API calls."""
import copy
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.reviews import ReviewJobs, create_reviews_router
from rag.generation import GenerationSettings
from rag.reviews.cards import card_input, validate_card
from rag.reviews.corpus import ReviewCorpus
from rag.reviews.engine import ReviewEngine, check_map, pack_evidence, plan_routes
from rag.reviews.schema import ReviewRequest
from rag.storage import atomic_json
from test_paper_graph import PaperGraphTests


def analysis(relevant=True, citations=None):
    return {'title':'Evidence-supported route','relevant':relevant,'insufficient_evidence':not relevant,
            'overview':[{'text':'Supported statement','citations':citations or ['S1']}] if relevant else [],
            'approaches':[],'differences':[],'experiments':[],'limitations':[],'missing_evidence':[]}


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = PaperGraphTests()
        cls.fixture.setUp()
        cls.fixture.build()
        cls.root = cls.fixture.root
        cls.corpus = ReviewCorpus(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def card(self):
        corpus = self.corpus
        payload, aliases = card_input(corpus, corpus.communities['topic'][0])
        def statement(members):
            ids = [next(a for a,c in aliases.items() if corpus.claims[c]['paper_id']==p) for p in members[:2]]
            return {'text':'Shared supported navigation clue','support':'multi_paper' if len(ids)>1 else 'paper_specific','claim_ids':ids}
        value = {'title':'Candidate topic','scope':[statement([p['paper_id'] for p in payload['papers']])],
                 'routes':[{'method_community_id':c['method_community_id'],'name':'Candidate method',
                     'overview':statement(c['paper_ids']),'differences':[],'experiments':[],'limitations':[],
                     'representative_paper_ids':c['paper_ids'][:2]} for c in payload['route_cells']]}
        return value,payload,aliases

    def test_card_support_requires_distinct_papers_and_known_claims(self):
        value,payload,aliases = self.card()
        checked = validate_card(value,self.corpus,payload,aliases)
        self.assertEqual(len(checked['scope'][0]['supporting_paper_ids']),2)
        bad = copy.deepcopy(value)
        bad['scope'][0]['claim_ids'] = [bad['scope'][0]['claim_ids'][0]]
        with self.assertRaisesRegex(ValueError,'distinct papers'):
            validate_card(bad,self.corpus,payload,aliases)
        bad['scope'][0]['claim_ids'] = ['unknown']
        with self.assertRaises(ValueError):
            validate_card(bad,self.corpus,payload,aliases)
        bad = copy.deepcopy(value)
        bad['routes'][0]['representative_paper_ids'] = ['outside']
        with self.assertRaisesRegex(ValueError,'Representative'):
            validate_card(bad,self.corpus,payload,aliases)

    def test_anchor_text_and_pdf_version_are_checked(self):
        corpus = self.corpus
        sid = next(iter(corpus.sources))
        source = corpus.source(sid)
        page = corpus.canonical_pages(source['paper_id'])[source['pdf_page']]
        self.assertEqual(source['text'],page['text'][source['source_start']:source['source_end']])
        path = self.root/corpus.documents[source['paper_id']]['pdf_path']
        original = path.read_bytes()
        try:
            path.write_bytes(b'changed PDF')
            with self.assertRaisesRegex(ValueError,'PDF'):
                corpus.source(sid)
        finally:
            path.write_bytes(original)

    def test_budget_dedup_and_route_paper_boundaries(self):
        claims = self.corpus.paper_claims('p0')
        route = {'selected_paper_ids':['p0']}
        registry = {}
        packed = pack_evidence(self.corpus,route,claims,[],10000,registry)
        self.assertEqual(len(packed['evidence']),1)
        self.assertEqual(len(packed['claim_links']),len(claims))
        self.assertLessEqual(packed['context_chars'],10000)
        empty = pack_evidence(self.corpus,route,claims,[],1,{})
        self.assertEqual(empty['evidence'],[])
        self.assertEqual(empty['missing_selected_paper_ids'],['p0'])
        with self.assertRaisesRegex(ValueError,'escaped'):
            pack_evidence(self.corpus,route,self.corpus.paper_claims('p1'),[],10000,{})

    def test_supplemental_hits_must_map_to_exact_canonical_text(self):
        source = self.corpus.source(self.corpus.paper_claims('p0')[0]['source_ids'][0])
        hit = {k:source[k] for k in ('paper_id','document_sha256','pdf_page','text')}
        hit['source_spans'] = [{'text_start':0,'text_end':len(source['text']),
            'source_start':source['source_start'],'source_end':source['source_end']}]
        self.assertEqual(self.corpus.hit_sources(hit)[0]['text'],source['text'])
        hit['text'] = 'a different source'
        with self.assertRaises(ValueError):
            self.corpus.hit_sources(hit)

    def test_map_rejects_unknown_citations_and_single_paper_comparisons(self):
        evidence = [{'id':'S1','paper_id':'p0'},{'id':'S2','paper_id':'p1'}]
        check_map(analysis(citations=['S1','S2']),evidence,['p0','p1'])
        with self.assertRaises(ValueError):
            check_map(analysis(citations=['S404']),evidence,['p0','p1'])
        value = analysis(citations=['S1','S2'])
        value['differences'] = [{'text':'A comparison','citations':['S1']}]
        with self.assertRaisesRegex(ValueError,'two different papers'):
            check_map(value,evidence,['p0','p1'])
        value['differences'][0]['citations'].append('S2')
        check_map(value,evidence,['p0','p1'])
        with self.assertRaisesRegex(ValueError,'each provided paper'):
            check_map(analysis(),evidence,['p0','p1'])
        value = analysis();value['missing_evidence'] = ['p1: evidence insufficient for this question']
        check_map(value,evidence,['p0','p1'])

    def test_library_plan_considers_every_topic_and_focus_stays_in_method(self):
        value,payload,aliases = self.card()
        cards = [validate_card(value,self.corpus,payload,aliases)]
        request = ReviewRequest(question='Research overview',scope='library')
        plan = plan_routes(self.corpus,cards,request,SimpleNamespace())
        self.assertEqual(set(plan['related_paper_ids']),set(self.corpus.papers))
        request.focus_route_ids = [plan['routes'][0]['route_id']]
        focus = plan_routes(self.corpus,cards,request,SimpleNamespace(),allowed_paper_ids=['p0'])
        self.assertEqual(focus['related_paper_ids'],['p0'])
        self.assertEqual(len(focus['routes']),1)
        request.focus_route_ids = ['unknown']
        with self.assertRaises(ValueError):
            plan_routes(self.corpus,cards,request,SimpleNamespace())

    def test_named_model_versions_need_support_in_the_cited_text(self):
        evidence = [{'id':'S1','paper_id':'p0','text':'We fine-tune a lightweight judge.'}]
        value = analysis()
        value['overview'][0]['text'] = 'The judge uses DeBERTa-v3-Large.'
        with self.assertRaisesRegex(ValueError,'Model/version'):
            check_map(value,evidence,['p0'])
        evidence[0]['text'] += ' The model is DeBERTa-v3-Large.'
        check_map(value,evidence,['p0'])

    def test_reduce_only_sees_supported_relevant_route_evidence(self):
        routes = [{'route_id':p,'name':p,'member_paper_ids':[p],'selected_paper_ids':[p],
                   'card_claim_ids':[],'method_community_id':'method','topic_community_ids':[],
                   'card_route_ids':[],'related_paper_ids':[p]} for p in ['p0','p1']]
        plan = {'routes':routes,'related_paper_ids':['p0','p1'],'topic_community_ids':[]}
        pipeline = SimpleNamespace(models=None,retrieve=lambda *a,**k:{'reranked_candidates':[]})
        def request_step(system,payload,settings,root,validator,purpose):
            if purpose=='scope_selection':
                value = {'selected_paper_ids':['p0','p1'],'reason':'Fixture scope'}
            elif purpose=='route_analysis':
                value = analysis(payload['candidate_route']=='p0')
            else:
                self.assertEqual({s['paper_id'] for s in payload['evidence']},{'p0'})
                value = {'title':'Sourced overview','overview':[{'text':'Result','citations':['S1']}],
                    'comparisons':[],'limitations':[],'covered_route_ids':['p0'],'insufficient_evidence':False}
            return validator(value),{'purpose':purpose,'cache_hit':False,'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        with patch('rag.reviews.engine.load_cards',return_value=({'cardset_id':'fixture','steps':[]},[])), \
             patch('rag.reviews.engine.plan_routes',return_value=plan), \
             patch('rag.reviews.engine.choose_claims',side_effect=lambda c,r,q,m:c.paper_claims(r['selected_paper_ids'][0])[:1]), \
             patch('rag.reviews.engine.generation_settings',return_value=GenerationSettings(api_key='test-only')), \
             patch('rag.reviews.engine.request_step',side_effect=request_step):
            result = ReviewEngine(self.root,self.corpus,pipeline).run({'question':'A research question'})
        self.assertEqual(result['synthesis']['covered_route_ids'],['p0'])
        self.assertEqual(len(result['evidence']),2)
        self.assertLessEqual(result['context_chars'],result['context_char_budget'])


class ReviewJobTests(unittest.TestCase):
    def test_web_job_can_be_enqueued_after_response_is_sent(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = ReviewJobs(Path(directory),lambda:None)
            tasks = []
            with patch('backend.reviews.build_cards',return_value={'cards':4}):
                job = jobs.start(kind='cards',enqueue=lambda *args:tasks.append(args))
                self.assertEqual(jobs.read(job['run_id'])['status'],'queued')
                self.assertEqual(len(tasks),1)
                tasks[0][0](*tasks[0][1:])
                jobs.executor.shutdown(wait=True)
            self.assertEqual(jobs.read(job['run_id'])['status'],'completed')

    def test_busy_job_persists_and_cannot_be_started_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = ReviewJobs(Path(directory),lambda:None)
            started,release = Event(),Event()
            def build(root,progress):
                started.set();release.wait(5);return {'cards':4}
            with patch('backend.reviews.build_cards',side_effect=build):
                job = jobs.start(kind='cards')
                self.assertTrue(started.wait(2))
                try:
                    with self.assertRaises(RuntimeError):
                        jobs.start(kind='cards')
                    self.assertEqual(jobs.read(job['run_id'])['status'],'running')
                    with self.assertRaises(KeyError):
                        jobs.read('../invalid')
                finally:
                    release.set();jobs.executor.shutdown(wait=True)
            self.assertEqual(jobs.read(job['run_id'])['result'],{'cards':4})

    def test_restart_marks_unfinished_job_and_bad_ids_are_404(self):
        from uuid import uuid4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid4())
            path = root/'runtime/reviews/jobs'/run_id/'job.json'
            atomic_json(path,{'run_id':run_id,'kind':'review','status':'running','created_at':'2026-01-01'})
            app = FastAPI();app.include_router(create_reviews_router(root,lambda:None))
            with TestClient(app) as client:
                self.assertEqual(client.get('/api/reviews/jobs/'+run_id).json()['status'],'interrupted')
                self.assertEqual(client.get('/api/reviews/jobs/not-a-uuid').status_code,404)
                self.assertEqual(client.post('/api/reviews',json={'question':' '}).status_code,422)


if __name__ == '__main__':
    unittest.main()

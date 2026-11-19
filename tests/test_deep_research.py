import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from rag.deep_research import EvidencePool, research
from rag.generation import GenerationSettings
from rag.query_planning_v2 import PaperRouterV2


A = 'ARES uses a human-labeled validation set for calibration.'
R = 'RAGAs evaluates faithfulness and answer relevance without reference answers.'
Q = 'RAGAs 和 ARES 评估什么？是否需要人工标注？'


def hit(paper, text, start=0):
    return {'paper_id': paper, 'title': paper, 'pdf_page': 3, 'text': text,
            'source_spans': [{'source_start': start, 'source_end': start + len(text)}]}


def module(paper, point, status, citations=()):
    return {'paper_id': paper, 'point': point, 'status': status, 'citations': list(citations)}


def search(target_module, query, *paper_ids):
    return {'module': target_module, 'query': query, 'scope': 'papers' if paper_ids else 'library',
            'paper_ids': list(paper_ids)}


def plan_for(query='human-labeled validation set'):
    return {'decision': 'supplement', 'modules': [module('r', '评估维度', 'supported', ['S1']),
            module('a', '人工验证样本', 'missing')], 'searches': [search(2, query, 'a')]}


class TestResearch(unittest.TestCase):
    def test_docling_offsets_only_deduplicate_within_the_same_paragraph(self):
        pool = EvidencePool()
        original = {'paper_id': 'r', 'pdf_page': 3, 'object_id': 'a', 'view_span': (0, 15),
                    'text': 'Original phrase'}
        contained = {**original, 'view_span': (0, 8), 'text': 'Original'}
        unrelated = {**contained, 'object_id': 'b'}
        self.assertTrue(pool.offer(original, 1600, 'initial'))
        self.assertFalse(pool.offer(contained, 1600, 'followup'))
        self.assertTrue(pool.offer(unrelated, 1600, 'followup'))

    def setUp(self):
        tmp = self.enterContext(TemporaryDirectory())
        root = Path(tmp)
        self.settings = GenerationSettings(api_key='test-key')
        papers = {'r': 'v1', 'a': 'v1', 'other': 'v1'}
        self.router = PaperRouterV2([{'paper_id': 'r', 'title': 'RAGAs'}, {'paper_id': 'a', 'title': 'ARES'}], papers)
        self.enterContext(patch('rag.deep_research.PaperRouterV2.from_project', return_value=self.router))
        self.queries, self.payloads = [], []
        self.pipeline = SimpleNamespace(root=root, settings=SimpleNamespace(corpus='papers', context_char_budget=1600),
                                        snapshot=lambda: ({'snapshot_id': 'index', 'documents': papers}, None),
                                        retrieve=self.retrieve)

    def retrieve(self, query, **kw):
        self.assertEqual(kw['auto_route'], not self.queries)
        scope = kw['paper_ids']
        self.queries.append((scope, query))
        if scope == ['r']:
            hits = [hit('r', R)]
        elif scope == ['a']:
            hits = [hit('a', 'ARES synthesizes the training data for its judges. ')]
            if 'calibrat' in query or 'human-labeled validation set' in query:
                hits.extend([hit('a', A, 200), hit('r', 'Wrong scope')])
        elif scope == ['r', 'a'] or query in (Q, '哪些工作评估生成质量？'):
            hits = [hit('r', R), hit('a', 'ARES synthesizes the training data for its judges. ')]
        else:
            hits = [hit('other', 'Another paper separately measures generation quality.')]
        return {'snapshot_id': 'index', 'reranked_candidates': hits}

    def run_research(self, replies, question=Q, **kw):
        replies = iter(replies)

        def respond(request):
            sent = json.loads(request.content)
            self.assertEqual(sent['response_format'], {'type': 'json_object'})
            self.payloads.append(json.loads(sent['messages'][1]['content']))
            next_reply = next(replies)
            content = json.dumps(next_reply, ensure_ascii=False) if isinstance(next_reply, dict) else next_reply
            return httpx.Response(200, json={'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                                             'usage': {'total_tokens': 20}})

        return research(self.pipeline, question, settings=self.settings, transport=httpx.MockTransport(respond), **kw)

    def test_scoped_followup_and_grounded_answer(self):
        final = {'findings': [{'module': 2, 'answer': 'ARES 使用人工标注的验证集做校准。',
                               'supports': ['S3.1']}], 'missing_modules': [1]}
        result = self.run_research([plan_for(), final])
        self.assertEqual([scope for scope, _ in self.queries], [None, ['a']])
        self.assertEqual(self.queries[-1][1], 'human-labeled validation set')
        self.assertEqual(result['research']['stop_reason'], 'followup_round_executed')
        self.assertEqual([hit['citation_id'] for hit in result['evidence']], ['S1', 'S2', 'S3'])
        self.assertEqual([s['id'] for s in self.payloads[0]['evidence']], ['S1', 'S2'])
        self.assertEqual([s['id'] for s in self.payloads[1]['evidence']], ['S1', 'S2', 'S3'])
        self.assertIn('[S3]', result['answer'])
        self.assertEqual(result['research']['findings'][0]['supports'][0]['quote'], A)
        self.assertEqual([m['module'] for m in result['missing_modules']], [1])
        self.assertEqual(result['usage']['total_tokens'], 40)

    def test_retrieval_only_runs_followup_without_synthesis(self):
        result = self.run_research([plan_for()], synthesize=False)
        self.assertEqual(self.queries, [(None, Q), (['a'], 'human-labeled validation set')])
        self.assertEqual([h['research_origin'] for h in result['evidence']], ['initial', 'initial', 'followup'])
        self.assertEqual(result['generation_status'], 'disabled')
        self.assertEqual([call['stage'] for call in result['research']['model_calls']], ['assessment'])

    def test_first_round_auto_scope_does_not_limit_later_library_search(self):
        retrieve = self.pipeline.retrieve

        def routed(query, **kwargs):
            result = retrieve(query, **kwargs)
            if kwargs['auto_route']:
                result['route'] = {'mode': 'single_paper', 'paper_ids': ['r']}
            return result

        self.pipeline.retrieve = routed
        plan = {'decision': 'supplement', 'modules': [module(None, '还有什么评估工作', 'partial', ['S1'])],
                'searches': [search(1, 'more work on evaluation measures')]}
        result = self.run_research([plan], synthesize=False)
        self.assertEqual([h['paper_id'] for h in result['initial_evidence']], ['r'])
        self.assertEqual([s['scope'] for s in result['research']['searches']], ['papers', 'library'])
        self.assertEqual([h['paper_id'] for h in result['evidence']], ['r', 'other'])

    def test_followup_displaces_only_unprotected_context_but_keeps_original_ids(self):
        self.pipeline.settings.context_char_budget = 420

        def result_for(query, **kwargs):
            if kwargs['auto_route']:
                hits = [hit('r', 'a' * 55), hit('r', 'b' * 55, 100), hit('a', 'c' * 55)]
            else:
                hits = [hit('a', 'new evidence ' + 'd' * 49, 200)]
            return {'snapshot_id': 'index', 'reranked_candidates': hits}

        self.pipeline.retrieve = result_for
        plan = {'decision': 'supplement', 'modules': [module('r', 'grounded claim', 'supported', ['S2.1']),
                                                       module('a', 'missing claim', 'missing')],
                'searches': [search(2, 'find original support', 'a')]}
        result = self.run_research([plan], synthesize=False)
        self.assertEqual([h['citation_id'] for h in result['initial_evidence']], ['S1', 'S2', 'S3'])
        self.assertIn('S2', {h['citation_id'] for h in result['evidence']})
        self.assertEqual(len(result['evidence']), 3)
        self.assertEqual(result['evidence'][-1]['citation_id'], 'S4')
        self.assertLessEqual(result['context_chars'], 420)

    def test_stop_when_both_papers_are_supported(self):
        plan = {'decision': 'stop', 'modules': [module('r', '评估维度', 'supported', ['S1']),
                module('a', '训练数据', 'supported', ['S2'])]}
        final = {'findings': [{'module': i, 'answer': '在原文支持范围内回答。',
                               'supports': [f'S{i}.1']} for i in (1, 2)], 'missing_modules': []}
        result = self.run_research([plan, final])
        self.assertEqual(self.queries, [(None, Q)])
        self.assertEqual(result['research']['stop_reason'], 'initial_evidence_sufficient')
        self.assertFalse(result['insufficient_evidence'])
        self.assertEqual(len(result['research']['model_calls']), 2)

    def test_bad_plan_cannot_trigger_unvalidated_searches(self):
        over_budget = plan_for()
        over_budget['searches'] *= 4
        foreign = plan_for()
        foreign['modules'][1]['paper_id'] = 'outside'
        result = self.run_research([over_budget, foreign])
        self.assertEqual(self.queries, [(None, Q)])
        self.assertIsNone(result['answer'])
        self.assertEqual(len(result['missing_modules']), 1)
        self.assertEqual(result['research']['stop_reason'], 'assessment_invalid_output')

    def test_final_keeps_valid_findings_and_reports_omitted_or_invalid_modules(self):
        final = {'findings': [{'module': 2, 'answer': '没有人工标注。',
                              'supports': ['S3.999']}],
                 'missing_modules': []}
        result = self.run_research([plan_for(), final])
        self.assertEqual(len(self.queries), 2)
        self.assertIsNone(result['answer'])
        self.assertEqual([m['module'] for m in result['missing_modules']], [1, 2])
        self.assertEqual([r['reason'] for r in result['research']['synthesis_repairs']],
                         ['omitted_module', 'finding_with_invalid_citations'])

    def test_search_hit_does_not_close_gap_or_trigger_another_round(self):
        result = self.run_research([plan_for(), {'findings': [{'module': 2, 'answer': '找到部分依据。',
                 'supports': ['S3.1']}], 'missing_modules': [1, 2]}])
        self.assertEqual(len(self.queries), 2)
        self.assertEqual([m['module'] for m in result['missing_modules']], [1, 2])
        self.assertTrue(result['insufficient_evidence'])

    def test_dedup_does_not_renumber_or_overfill_evidence(self):
        pool = EvidencePool()
        first = hit('a', A, 200)
        pool.offer(first, 360, 'initial')
        pool.offer(first, 360, 'followup', 1)
        pool.offer(hit('a', A[10:30], 210), 360, 'followup', 1)
        pool.offer(hit('r', 'z' * 500), 360, 'followup', 1)
        pool.offer(hit('r', R), 360, 'followup', 2)
        self.assertEqual([h['citation_id'] for h in pool.hits], ['S1', 'S2'])
        self.assertEqual([h['research_origin'] for h in pool.hits], ['initial', 'followup'])
        self.assertLessEqual(pool.used, 360)

    def test_open_question_can_discover_a_new_paper_with_a_library_followup(self):
        question = '哪些工作评估生成质量？'
        query = 'measures generation quality'
        plan = {'decision': 'supplement', 'modules': [module(None, '有哪些评估工作', 'partial', ['S1.1'])],
                'searches': [search(1, query)]}
        final = {'findings': [{'module': 1, 'answer': '两篇工作提供了不同评估方式。',
                               'supports': ['S1.1', 'S3.1']}], 'missing_modules': []}
        result = self.run_research([plan, final], question=question)
        self.assertEqual(self.queries, [(None, question), (None, query)])
        self.assertEqual(result['citations'], ['S1', 'S3'])
        self.assertEqual(result['evidence'][-1]['paper_id'], 'other')

    def test_assessment_drops_nonexistent_excerpt_and_preserves_valid_search(self):
        plan = plan_for()
        plan['modules'][0]['citations'] = ['S1.999']
        result = self.run_research([plan, {'findings': [], 'missing_modules': [1, 2]}])
        self.assertEqual(self.queries, [(None, Q), (['a'], 'human-labeled validation set')])
        row = result['research']['modules'][0]
        self.assertEqual((row['status'], row['citations']), ('missing', []))
        self.assertEqual(result['research']['assessment_repairs'][0]['discarded_citations'], ['S1.999'])

    def test_assessment_downgrades_mixed_paper_citations_without_fabricating_a_search(self):
        plan = {'decision': 'stop', 'modules': [module('a', '训练数据', 'supported', ['S1.1', 'S2.1', 'S2.1'])]}
        result = self.run_research([plan, {'findings': [], 'missing_modules': [1]}])
        self.assertEqual(self.queries, [(None, Q)])
        row = result['research']['modules'][0]
        self.assertEqual((row['paper_id'], row['status'], row['citations']), ('a', 'partial', ['S2.1']))
        self.assertEqual(result['research']['stop_reason'], 'no_followup_planned')

    def test_supplement_missing_searches_or_paper_ids_cannot_execute(self):
        missing_searches = plan_for()
        del missing_searches['searches']
        missing_ids = plan_for()
        del missing_ids['searches'][0]['paper_ids']
        result = self.run_research([missing_searches, missing_ids])
        self.assertEqual(self.queries, [(None, Q)])
        self.assertEqual(result['research']['stop_reason'], 'assessment_invalid_output')

    def test_safe_empty_fields_for_missing_evidence_and_library_scope(self):
        plan = {'decision': 'supplement', 'modules': [{'paper_id': None, 'point': '评估方式', 'status': 'missing'}],
                'searches': [{'module': 1, 'query': 'generation quality', 'scope': 'library'}]}
        result = self.run_research([plan, {'findings': [], 'missing_modules': [1]}])
        self.assertEqual(self.queries, [(None, Q), (None, 'generation quality')])
        self.assertEqual(result['research']['modules'][0]['citations'], [])

    def test_assessment_repair_does_not_relax_final_citation_ownership(self):
        plan = {'decision': 'stop', 'modules': [module('a', '训练数据', 'supported', ['S1.1']),
                                                 module('r', '评估维度', 'supported', ['S1.1'])]}
        final = {'findings': [{'module': 1, 'answer': '另一篇的结果。', 'supports': ['S2.1', 'S1.1']},
                              {'module': 2, 'answer': '评估忠实性。', 'supports': ['S1.1']}], 'missing_modules': []}
        result = self.run_research([plan, final])
        self.assertEqual(self.payloads[1]['excerpt_ids_by_paper'], {'r': ['S1.1'], 'a': ['S2.1']})
        self.assertEqual(result['citations'], ['S1'])
        self.assertNotIn('另一篇的结果', result['answer'])
        self.assertEqual([m['module'] for m in result['missing_modules']], [1])

    def test_agent_can_search_a_subset_of_multiple_papers(self):
        query = 'evaluation measures'
        plan = {'decision': 'supplement', 'modules': [module(None, '评估方式', 'missing')],
                'searches': [search(1, query, 'r', 'a')]}
        result = self.run_research([plan, {'findings': [], 'missing_modules': [1]}])
        self.assertEqual(self.queries, [(None, Q), (['r', 'a'], query)])
        self.assertEqual(result['research']['searches'][1]['paper_ids'], ['r', 'a'])

    def test_one_selected_paper_remains_the_boundary_for_library_followup(self):
        query = 'faithfulness metrics'
        plan = {'decision': 'supplement', 'modules': [module(None, '评估范围', 'missing')],
                'searches': [search(1, query)]}
        result = self.run_research([plan, {'findings': [], 'missing_modules': [1]}], paper_ids=['r'])
        self.assertEqual(self.queries, [(['r'], Q), (['r'], query)])
        self.assertEqual(result['research']['selected_paper_ids'], ['r'])

    def test_invalid_or_manual_out_of_scope_target_is_never_searched(self):
        plan = {'decision': 'supplement', 'modules': [module(None, '评估范围', 'missing')],
                'searches': [search(1, 'evaluation measures', 'other')]}
        unindexed = {**plan, 'searches': [search(1, 'evaluation measures', 'unknown')]}
        result = self.run_research([plan, unindexed], paper_ids=['r'])
        self.assertEqual(self.queries, [(['r'], Q)])
        self.assertEqual(result['research']['stop_reason'], 'assessment_invalid_output')

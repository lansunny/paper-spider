import unittest
from unittest.mock import Mock

from rag.routing import PaperRouter
from scripts.experiment_routing import retrieve_branches


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {'paper_id': 'r', 'title': 'RAGAs: Automated Evaluation of Retrieval Augmented Generation'},
            {'paper_id': 'a', 'title': 'ARES: An Automated Evaluation Framework'},
            {'paper_id': 'd', 'title': 'Dense Passage Retrieval for Open-Domain Question Answering'},
            {'paper_id': 'q', 'title': 'RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering'},
            {'paper_id': 'i', 'title': 'Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions'}]
        self.router = PaperRouter(self.catalog, ['r', 'a', 'd', 'q', 'i'], {'d': ['DPR'], 'i': ['IRCoT']})

    def test_single_paper_and_internal_objects(self):
        for question in ['根据 RAGAs 原论文，如何从生成回答和检索上下文计算 Faithfulness？',
                         '《RAGAs》如何计算 Faithfulness？', 'How does RAGAs evaluate faithfulness?']:
            with self.subTest(question=question):
                result = self.router.route(question)
                self.assertEqual(result['mode'], 'single_paper')
                self.assertEqual(result['paper_ids'], ['r'])

    def test_comparisons_open_queries_and_unknown_second_paper_stay_global(self):
        for question in ['RAGAs 与 ARES 如何评估？', 'RAGAs 和 UnseenMethod 的实验结果',
                         'DPR 比较了哪三类负例？', '有没有类似 RAGAs 的评估方法？',
                         '如何衡量回答忠实度？', 'RAGAs 以及《Unknown paper》的结论',
                         'Compare DPR against an unknown method', 'ARES 和某篇新论文的结果']:
            with self.subTest(question=question):
                self.assertEqual(self.router.route(question)['mode'], 'global')

    def test_alias_collision_and_missing_index_abstain(self):
        ambiguous = PaperRouter(self.catalog, ['r', 'a'], {'r': ['ABC'], 'a': ['ABC']})
        self.assertEqual(ambiguous.route('ABC 如何评估？')['reason'], 'multiple_or_ambiguous_papers')
        unavailable = PaperRouter(self.catalog, ['a'])
        self.assertEqual(unavailable.route('RAGAs 如何评估？')['reason'], 'paper_not_in_snapshot')

    def test_longest_title_and_ascii_boundaries(self):
        result = self.router.route(self.catalog[3]['title']+' 的训练方法是什么？')
        self.assertEqual(result['paper_ids'], ['q'])
        self.assertEqual(self.router.route('shares results')['mode'], 'global')
        self.assertEqual(self.router.route('DPRv2 如何训练？')['mode'], 'global')
        self.assertEqual(self.router.route('IRCoT 如何停止检索？')['paper_ids'], ['i'])

    def test_scope_is_sent_to_both_retrievers_before_top_k(self):
        lexical, store = Mock(), Mock()
        lexical.search.return_value = [{'paper_id': 'r'}]
        store.search.return_value = [{'paper_id': 'r'}]
        question = {'question': 'RAGAs 如何计算？', 'question_en': 'How does RAGAs compute?',
                    'gold_paper_ids': ['must-not-be-used']}
        route = self.router.route(question['question'])
        retrieve_branches(lexical, store, [1.0], 'snapshot', question, route)
        lexical.search.assert_called_once_with(question['question_en'], top_k=100, paper_ids=['r'])
        store.search.assert_called_once_with([1.0], 'snapshot', top_k=100, paper_ids=['r'], exact=False)

    def test_global_route_does_not_filter(self):
        lexical, store = Mock(), Mock()
        lexical.search.return_value = store.search.return_value = []
        question = {'question_en': 'Compare RAGAs and ARES'}
        retrieve_branches(lexical, store, [1.0], 'snapshot', question,
                          self.router.route(question['question_en']))
        self.assertIsNone(lexical.search.call_args.kwargs['paper_ids'])
        self.assertIsNone(store.search.call_args.kwargs['paper_ids'])

    def test_scope_violation_fails_explicitly(self):
        lexical, store = Mock(), Mock()
        lexical.search.return_value = []
        store.search.return_value = [{'paper_id': 'wrong'}]
        with self.assertRaisesRegex(ValueError, 'outside the routed scope'):
            retrieve_branches(lexical, store, [], 'snapshot', {'question_en': 'RAGAs'},
                              self.router.route('RAGAs 如何计算？'))


if __name__ == '__main__':
    unittest.main()

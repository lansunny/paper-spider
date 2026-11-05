"""Fixed-corpus evaluation guards and controls, independent of local models."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rag.graph_evaluation import (ScopedVectorStore, clustered_interval, matched_random_planner,
                                  paper_clusters, select_cases)
from rag.paper_graph.retrieval import GraphRetrievalOptions


def question(identifier, papers, answerable=True):
    return {'question_id': identifier, 'answerable': answerable,
        'evidence_units': [{'alternatives': [[{'paper_id': pid} for pid in papers]]}] if papers else []}


class GraphEvaluationTests(unittest.TestCase):
    def test_only_questions_fully_grounded_in_the_fixed_corpus_are_selected(self):
        rows = [question('inside', ['a']), question('outside', ['a','z']), question('unknown', [], False)]
        selected, excluded = select_cases({'fixture': rows}, ['a','b'])
        self.assertEqual([c['question']['question_id'] for c in selected], ['inside'])
        self.assertEqual({r['question_id'] for r in excluded}, {'outside','unknown'})
        self.assertEqual(excluded[0]['outside_paper_ids'], ['z'])

    def test_all_dense_queries_have_the_same_outer_corpus_scope(self):
        calls = []
        class Store:
            def search(self, vector, snapshot, limit, paper_ids, exact):
                calls.append(paper_ids)
                return [{'paper_id': p} for p in paper_ids]
        store = ScopedVectorStore(Store(), ['a','b'], 'snapshot')
        self.assertEqual(store.search([], 'snapshot'), [{'paper_id':'a'},{'paper_id':'b'}])
        self.assertEqual(store.search([], 'snapshot', paper_ids=['b','outside']), [{'paper_id':'b'}])
        self.assertEqual(store.search([], 'snapshot', paper_ids=['outside']), [])
        self.assertEqual(calls, [['a','b'], ['b']])
        with self.assertRaises(ValueError):
            store.search([], 'changed-snapshot')

    def test_out_of_scope_dense_results_fail_instead_of_entering_metrics(self):
        class Store:
            def search(self, *args): return [{'paper_id':'outside'}]
        with self.assertRaises(ValueError):
            ScopedVectorStore(Store(), ['a'], 'snapshot').search([], 'snapshot')

    def test_questions_sharing_papers_are_clustered_for_uncertainty(self):
        cases = [{'question': question('one', ['a'])}, {'question': question('two', ['a','b'])},
                 {'question': question('three', ['b'])}, {'question': question('four', ['c'])}]
        groups = paper_clusters(cases)
        self.assertEqual(groups['one'], groups['three'])
        self.assertNotEqual(groups['one'], groups['four'])
        interval = clustered_interval([0,.5,-.5,0], [groups[c['question']['question_id']] for c in cases])
        self.assertIsNone(interval['ci95'])
        self.assertFalse(interval['reliable'])

    def test_cluster_bootstrap_is_reproducible_and_marks_small_samples(self):
        a = clustered_interval([0,.5,-.5,.25], ['a','b','c','d'], seed=17)
        self.assertEqual(a, clustered_interval([0,.5,-.5,.25], ['a','b','c','d'], seed=17))
        self.assertEqual(a['paper_groups'], 4)
        self.assertFalse(a['reliable'])

    def test_random_control_matches_target_count_without_fabricating_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'catalog.sqlite3'
            with sqlite3.connect(path) as connection:
                connection.execute('CREATE TABLE papers(paper_id,title,document_sha256,profile_sha256)')
                for pid in 'abcd':
                    connection.execute('INSERT INTO papers VALUES(?,?,?,?)',(pid,pid,pid*64,'profile-'+pid))
            connection.close()
            class Store:
                def bundle(self): return Path(temporary), {'build_id':'fixed'}
                def _connection(self, bundle):
                    connection = sqlite3.connect(path)
                    connection.row_factory = sqlite3.Row
                    return connection
            trace = {'graph_build_id':'fixed', 'expanded_paper_ids':['b','c'], 'dimension':'method',
                     'seeds':[{'paper_id':'a'}], 'paths':[]}
            context = {'question':'compare methods', 'paper_ids':None,
                       'manifest':{'documents':{p:p*64 for p in 'abcd'}}}
            with patch('rag.graph_evaluation.plan_expansion', return_value=trace):
                first = matched_random_planner(42)(Store(), [], context, GraphRetrievalOptions(enabled=True))
                second = matched_random_planner(42)(Store(), [], context, GraphRetrievalOptions(enabled=True))
            self.assertEqual(first,second)
            self.assertEqual(len(first['expanded_paper_ids']),2)
            self.assertNotIn('a',first['expanded_paper_ids'])
            self.assertTrue(all(p['weight'] is None and p['not_a_graph_edge'] for p in first['paths']))


if __name__ == '__main__': unittest.main()

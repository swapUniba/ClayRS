from unittest import TestCase

import pandas as pd
from orange_cb_recsys.recsys.graphs import NXBipartiteGraph


class TestNXBipartiteGraph(TestCase):
    def test_all(self):
        df = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "aaa", "002", "003", "004", "004"],
                                     'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
                                     'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        # Create graph and populate it with data from dataframe
        g = NXBipartiteGraph(df)

        # Simple assert just to make sure the graph is created
        self.assertGreater(len(g.from_nodes), 0)
        self.assertGreater(len(g.to_nodes), 0)

        # Add 'from' node
        self.assertFalse(g.is_from_node('000'))
        g.add_from_node('000')
        self.assertTrue(g.is_from_node('000'))

        # Add 'to' node
        self.assertFalse(g.is_to_node('zzz'))
        g.add_to_node('zzz')
        self.assertTrue(g.is_to_node('zzz'))

        # Link existent 'from' node and existent 'to' node
        self.assertIsNone(g.get_link_data('000', 'aaa'))
        g.link_from_to('000', 'aaa', 0.5)
        self.assertIsNotNone(g.get_link_data('000', 'aaa'))

        # Link non-existent 'from' node and non-existent 'to' node,
        # so both nodes will be created
        self.assertFalse(g.is_from_node('u1'))
        self.assertFalse(g.is_to_node('Tenet'))
        self.assertIsNone(g.get_link_data('u1', 'Tenet'))
        g.link_from_to('u1', 'Tenet', 0.5)
        self.assertTrue(g.is_from_node('u1'))
        self.assertTrue(g.is_to_node('Tenet'))
        self.assertIsNotNone(g.get_link_data('000', 'aaa'))

        # Get predecessors of a node
        result = g.get_predecessors('Tenet')
        expected = ['u1']
        self.assertEqual(expected, result)

        # Get successors of a node
        result = g.get_successors('u1')
        expected = ['Tenet']
        self.assertEqual(expected, result)

        # Get all voted contents of a node
        result = g.get_voted_contents('u1')
        expected = ['Tenet']
        self.assertEqual(expected, result)

from unittest import TestCase

import pandas as pd
from orange_cb_recsys.recsys.graphs import NXBipartiteGraph
import networkx as nx


class TestNXBipartiteGraph(TestCase):

    def setUp(self) -> None:
        df = pd.DataFrame.from_dict({'from_id': ["u1", "u1", "u2", "u2", "u2", "u3", "u4", "u4"],
                                     'to_id': ["i1", "i2", "i1", "i3", "i4", "i4", "i3", "i4"],
                                     'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})
        # Graph that will be used by all tests
        self.g: NXBipartiteGraph = NXBipartiteGraph(df)

    def test_graph_created(self):
        # Simple assert just to make sure the graph is created
        self.assertGreater(len(self.g.user_nodes), 0)
        self.assertGreater(len(self.g.item_nodes), 0)

    def test_add_user(self):
        # Add 'user' node
        self.assertFalse(self.g.is_user_node('u0'))
        self.g.add_user_node('u0')
        self.assertTrue(self.g.is_user_node('u0'))

        # Add 'user' node but it already exists as
        # an 'item' node, so it exists as both
        self.assertTrue(self.g.is_item_node('i1'))
        self.g.add_user_node('i1')
        self.assertTrue(self.g.is_user_node('i1'))
        self.assertTrue(self.g.is_item_node('i1'))

    def test_add_item(self):
        # Add 'item' node
        self.assertFalse(self.g.is_item_node('i0'))
        self.g.add_item_node('i0')
        self.assertTrue(self.g.is_item_node('i0'))

        # Add 'item' node but it already exists as
        # a 'user' node, so it exists as both
        self.assertTrue(self.g.is_user_node('u1'))
        self.g.add_item_node('u1')
        self.assertTrue(self.g.is_item_node('u1'))
        self.assertTrue(self.g.is_user_node('u1'))

    def test_add_link_user_item(self):
        # Link existent 'user' node to an existent 'item' node
        self.g.add_user_node('u0')
        self.g.add_item_node('Tenet')
        self.assertIsNone(self.g.get_link_data('u0', 'Tenet'))
        self.g.add_link('u0', 'Tenet')
        expected = {'label': 'score_label', 'weight': 0.5}
        result = self.g.get_link_data('u0', 'Tenet')
        self.assertEqual(expected, result)

        # Link existent 'item' node to an existent 'user' node
        self.g.add_item_node('Tenet')
        self.g.add_user_node('1')
        self.assertIsNone(self.g.get_link_data('Tenet', '1'))
        self.g.add_link('Tenet', '1', 0.5)
        self.assertFalse(self.g.is_user_node('Tenet'))
        self.assertFalse(self.g.is_item_node('1'))
        self.assertIsNotNone(self.g.get_link_data('Tenet', '1'))

        # Try to Link non-existent 'user' node and non-existent 'item' node,
        # so no link is created
        self.assertFalse(self.g.node_exists('u_new'))
        self.assertFalse(self.g.node_exists('i_new'))
        self.g.add_link('u_new', 'i_new', 0.5)
        self.assertFalse(self.g.is_user_node('u_new'))
        self.assertFalse(self.g.is_item_node('i_new'))
        self.assertIsNone(self.g.get_link_data('u_new', 'i_new'))

    def test_pred_succ(self):
        # Get predecessors of a node
        self.g.add_user_node('u0')
        self.g.add_user_node('u1')
        self.g.add_item_node('i0')
        self.g.add_link('u0', 'i0', 0.5)
        self.g.add_link('u1', 'i0', 0.5)
        result = self.g.get_predecessors('i0')
        expected = ['u0', 'u1']
        self.assertEqual(expected, result)

        # Get successors of a node
        self.g.add_user_node('u0')
        self.g.add_item_node('i0')
        self.g.add_item_node('i1')
        self.g.add_link('u0', 'i0', 0.5)
        self.g.add_link('u0', 'i1', 0.5)
        result = self.g.get_successors('u0')
        expected = ['i0', 'i1']
        self.assertEqual(expected, result)

        # Get voted contents of a node
        result = self.g.get_voted_contents('u0')
        expected = ['i0', 'i1']
        self.assertEqual(expected, result)

    def test__graph(self):
        # Simple assert just to test the _graph method
        self.assertIsInstance(self.g._graph, nx.DiGraph)

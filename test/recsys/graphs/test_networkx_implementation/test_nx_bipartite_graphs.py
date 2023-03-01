import shutil
from unittest import TestCase

import pandas as pd
import os
import lzma
import pickle

from clayrs.content_analyzer.ratings_manager.ratings import Ratings
from clayrs.recsys.graphs.nx_implementation.nx_bipartite_graphs import NXBipartiteGraph
import networkx as nx

from clayrs.recsys.graphs.graph import UserNode, ItemNode, PropertyNode

rat = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                              'to_id': ["tt0112281", "tt0112302", "tt0112281", "tt0112346",
                                        "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                              'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})
rat = Ratings.from_dataframe(rat)

rat_timestamp = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                        'to_id': ["tt0112281", "tt0112302", "tt0112281", "tt0112346",
                                                  "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                                        'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7],
                                        'timestamp': [11111, 22222, 33333, 44444,
                                                      55555, 66666, 777777, 88888]
                                        })
rat_timestamp = Ratings.from_dataframe(rat_timestamp, timestamp_column='timestamp')


class TestNXBipartiteGraph(TestCase):

    def setUp(self) -> None:
        # graphs that will be used for testing
        self.g: NXBipartiteGraph = NXBipartiteGraph(rat)

        self.graph_custom_label: NXBipartiteGraph = NXBipartiteGraph(rat, link_label='my_label')

        self.graph_timestamp: NXBipartiteGraph = NXBipartiteGraph(rat_timestamp)

        self.empty_graph: NXBipartiteGraph = NXBipartiteGraph()

    def test_graph_creation(self):
        # test that all nodes are added to the graph

        user_column = rat.user_id_column
        item_column = rat.item_id_column
        score_column = rat.score_column

        ratings_iterator = zip(user_column, item_column, score_column)

        for interaction in ratings_iterator:
            self.assertTrue(UserNode(interaction[0]) in self.g.user_nodes)
            self.assertTrue(ItemNode(interaction[1]) in self.g.item_nodes)

            link_data = self.g.get_link_data(UserNode(interaction[0]), ItemNode(interaction[1]))
            self.assertIsNotNone(link_data)

            expected_data_link = {'weight': interaction[2]}
            result = self.g.get_link_data(UserNode(interaction[0]), ItemNode(interaction[1]))

            self.assertEqual(expected_data_link, result)

    def test_graph_creation_custom_label(self):
        # test that all nodes are added to the graph and the link between them has
        # a changed label

        user_column = rat.user_id_column
        item_column = rat.item_id_column
        score_column = rat.score_column

        ratings_iterator = zip(user_column, item_column, score_column)

        for interaction in ratings_iterator:
            self.assertTrue(UserNode(interaction[0]) in self.graph_custom_label.user_nodes)
            self.assertTrue(ItemNode(interaction[1]) in self.graph_custom_label.item_nodes)

            link_data = self.graph_custom_label.get_link_data(UserNode(interaction[0]),
                                                              ItemNode(interaction[1]))
            self.assertIsNotNone(link_data)

            expected_data_link = {'label': 'my_label', 'weight': interaction[2]}
            result = self.graph_custom_label.get_link_data(UserNode(interaction[0]), ItemNode(interaction[1]))

            self.assertEqual(expected_data_link, result)

    def test_graph_creation_w_timestamp(self):

        user_column = rat.user_id_column
        item_column = rat.item_id_column
        score_column = rat.score_column
        timestamp_column = rat.timestamp_column

        ratings_iterator = zip(user_column, item_column, score_column, timestamp_column)

        for interaction in ratings_iterator:
            expected_data_link = {'weight': interaction[2], 'timestamp': interaction[3]}
            result = self.graph_timestamp.get_link_data(UserNode(interaction[0]), ItemNode(interaction[1]))

            self.assertEqual(expected_data_link, result)

    def test_graph_creation_empty(self):
        self.assertTrue(len(self.empty_graph.user_nodes) == 0)
        self.assertTrue(len(self.empty_graph.item_nodes) == 0)

    def test_add_user(self):
        # Add 'user' node
        self.assertFalse(UserNode('u0') in self.g.user_nodes)
        self.assertFalse(self.g.node_exists(UserNode('u0')))
        self.g.add_node(UserNode('u0'))
        self.assertTrue(self.g.node_exists(UserNode('u0')))
        self.assertTrue(UserNode('u0') in self.g.user_nodes)

        # Add a list of 'user' nodes
        list_nodes = [UserNode('u1_list'), UserNode('u2_list'), UserNode('u3_list')]
        self.g.add_node(list_nodes)
        for n in list_nodes:
            self.assertTrue(self.g.node_exists(n))
            self.assertTrue(n in self.g.user_nodes)

        # Add 'user' node with same id of an already existent
        # 'item' node, verify that both exists as different entities
        self.g.add_node(ItemNode('i0'))
        self.g.node_exists(ItemNode('i0'))
        self.assertTrue(ItemNode('i0') in self.g.item_nodes)
        self.g.add_node(UserNode('i0'))
        self.assertTrue(self.g.node_exists(UserNode('i0')))
        self.assertTrue(self.g.node_exists(ItemNode('i0')))
        self.assertTrue(UserNode('i0') in self.g.user_nodes)
        self.assertTrue(ItemNode('i0') in self.g.item_nodes)

    def test_add_item(self):
        # Add 'item' node
        self.assertFalse(ItemNode('i0') in self.g.item_nodes)
        self.g.add_node(ItemNode('i0'))
        self.assertTrue(self.g.node_exists(ItemNode('i0')))
        self.assertTrue(ItemNode('i0') in self.g.item_nodes)

        # Add a list of 'item' nodes
        list_nodes = [ItemNode('i1_list'), ItemNode('i2_list'), ItemNode('i3_list')]
        self.g.add_node(list_nodes)
        for n in list_nodes:
            self.assertTrue(self.g.node_exists(n))
            self.assertTrue(n in self.g.item_nodes)

        # Add 'item' node with same id of an already existent
        # 'user' node, verify that both exists as different entities
        self.g.add_node(UserNode('u0'))
        self.assertTrue(self.g.node_exists(UserNode('u0')))
        self.assertTrue(UserNode('u0') in self.g.user_nodes)
        self.g.add_node(ItemNode('u0'))
        self.assertTrue(self.g.node_exists(UserNode('u0')))
        self.assertTrue(self.g.node_exists(ItemNode('u0')))
        self.assertTrue(UserNode('u0') in self.g.user_nodes)
        self.assertTrue(ItemNode('u0') in self.g.item_nodes)

    def test_add_raise_error(self):

        # test add a node which is not item node nor user node
        with self.assertRaises(ValueError):
            self.g.add_node(PropertyNode('prop'))

        # test add a list of node in which a node is not a user or item node
        with self.assertRaises(ValueError):
            self.g.add_node([UserNode('u'), ItemNode('i'), PropertyNode('p')])

    def test_add_link_user_item_existent(self):
        # Link existent 'user' node to an existent 'item' node
        user_node = UserNode('u0')
        item_node = ItemNode('Tenet')
        self.g.add_node(user_node)
        self.g.add_node(item_node)
        self.assertIsNone(self.g.get_link_data(user_node, item_node))
        self.g.add_link(user_node, item_node, 0.5)
        expected = {'weight': 0.5}
        result = self.g.get_link_data(user_node, item_node)
        self.assertEqual(expected, result)

        # Link list of existent 'user' nodes to a list of existent 'item' nodes
        user_list = [UserNode('u1_list'), UserNode('u1_list'), UserNode('u3_list')]
        self.g.add_node(user_list)
        items_list = [ItemNode('i1_list'), ItemNode('i2_list'), ItemNode('i3_list')]
        self.g.add_node(items_list)
        self.g.add_link(user_list, items_list, 0.5)
        for user, item in zip(user_list, items_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

        # Link existent 'item' node to an existent 'user' node
        user_node = UserNode('1')
        item_node = ItemNode('Tenet')
        self.g.add_node(item_node)
        self.g.add_node(user_node)
        self.assertIsNone(self.g.get_link_data(item_node, user_node))
        self.g.add_link(item_node, user_node, 0.5)
        self.assertFalse(self.g.node_exists(UserNode('Tenet')))
        self.assertFalse(self.g.node_exists(ItemNode('u1')))
        self.assertIsNotNone(self.g.get_link_data(item_node, user_node))

        # Link list of existent 'item' nodes to a list of existent 'user' nodes
        items_list = [ItemNode('i1_list'), ItemNode('i2_list'), ItemNode('i3_list')]
        self.g.add_node(items_list)
        users_list = [UserNode('u1_list'), UserNode('u2_list'), UserNode('u3_list')]
        self.g.add_node(users_list)
        self.g.add_link(items_list, users_list, 0.5)
        for item, user in zip(items_list, users_list):
            result = self.g.get_link_data(item, user)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    def test_add_link_user_item_non_existent(self):

        # Link non-existent 'user' node and non-existent 'item' node,
        # so both nodes are created and then linked
        user_new = UserNode('u_new')
        item_new = ItemNode('i_new')
        self.assertFalse(self.g.node_exists(user_new))
        self.assertFalse(self.g.node_exists(item_new))
        self.g.add_link(user_new, item_new, 0.5)
        self.assertTrue(self.g.node_exists(user_new))
        self.assertTrue(self.g.node_exists(item_new))
        self.assertIsNotNone(self.g.get_link_data(user_new, item_new))

        # Link non-existent 'user' node list and non-existent 'item' node list,
        # so all nodes of the two lists are created and then linked
        user_new_list = [UserNode('u_new_new1'), UserNode('u_new_new2')]
        item_new_list = [ItemNode('i_new_new1'), ItemNode('i_new_new2')]
        for user in user_new_list:
            self.assertFalse(self.g.node_exists(user))
        for item in item_new_list:
            self.assertFalse(self.g.node_exists(item))

        self.g.add_link(user_new_list, item_new_list, 0.5)

        for user, item in zip(user_new_list, item_new_list):
            result = self.g.get_link_data(user, item)
            expected = {'weight': 0.5}

            self.assertEqual(expected, result)

    def test_add_link_raise_error(self):
        # test link property node in a bipartite graph
        with self.assertRaises(ValueError):
            self.g.add_link(PropertyNode('prop'), ItemNode('item'))

    def test_add_link_custom_parameters(self):
        # Link non-existent 'user' node and non-existent 'item' node
        # with no parameters
        user_new = UserNode('u_new')
        item_new = ItemNode('i_new')
        self.assertFalse(self.g.node_exists(user_new))
        self.assertFalse(self.g.node_exists(item_new))
        self.g.add_link(user_new, item_new, label=None)
        self.assertTrue(self.g.node_exists(user_new))
        self.assertTrue(self.g.node_exists(item_new))
        expected = {}
        result = self.g.get_link_data(user_new, item_new)

        self.assertEqual(expected, result)

        # Link non-existent 'user' node and non-existent 'item' node
        # with all custom parameters
        user_new = UserNode('u_new1')
        item_new = ItemNode('i_new1')
        self.assertFalse(self.g.node_exists(user_new))
        self.assertFalse(self.g.node_exists(item_new))
        self.g.add_link(user_new, item_new, weight=0.5, label='my_label', timestamp='001122')
        self.assertTrue(self.g.node_exists(user_new))
        self.assertTrue(self.g.node_exists(item_new))
        expected = {'label': 'my_label', 'weight': 0.5, 'timestamp': '001122'}
        result = self.g.get_link_data(user_new, item_new)

        self.assertEqual(expected, result)

    def test_remove_link(self):
        # test remove link between existent nodes
        user_new = UserNode('u_new')
        item_new = ItemNode('i_new')
        self.g.add_link(user_new, item_new, label='test_link')

        # test that the link is present
        expected = {'label': 'test_link'}
        result = self.g.get_link_data(user_new, item_new)
        self.assertEqual(expected, result)

        # test that the link is removed
        self.g.remove_link(user_new, item_new)
        self.assertIsNone(self.g.get_link_data(user_new, item_new))

        # test remove link between non existent nodes
        result = self.g.remove_link(UserNode('non_existent'), ItemNode('non_existent1'))
        self.assertIsNone(result)

    def test_pred_succ(self):
        # Get predecessors of a node
        self.g.add_link(UserNode('u0'), ItemNode('i0'), 0.5)
        self.g.add_link(UserNode('u1'), ItemNode('i0'), 0.5)
        result = self.g.get_predecessors(ItemNode('i0'))
        expected = [UserNode('u0'), UserNode('u1')]
        self.assertEqual(expected, result)

        # Get predecessors of a non-existent node
        with self.assertRaises(TypeError):
            self.g.get_predecessors(ItemNode('non_existent'))

        # Get successors of a node
        self.g.add_link(UserNode('u0'), ItemNode('i0'), 0.5)
        self.g.add_link(UserNode('u0'), ItemNode('i1'), 0.5)
        result = self.g.get_successors(UserNode('u0'))
        expected = [ItemNode('i0'), ItemNode('i1')]
        self.assertEqual(expected, result)

        # Get successors of a non-existent node
        with self.assertRaises(TypeError):
            self.g.get_successors(UserNode('non_existent'))

    def test_metrics(self):
        # We calculate some metrics, simple assert to make sure they are
        # calculated
        self.assertGreater(len(self.g.degree_centrality()), 0)
        self.assertGreater(len(self.g.closeness_centrality()), 0)
        self.assertGreater(len(self.g.dispersion()), 0)

    def test_to_networkx(self):
        # Simple assert just to test the _graph method
        self.assertIsInstance(self.g.to_networkx(), nx.DiGraph)

    def test_remove_node(self):
        # remove a single node
        node = UserNode("to_remove")
        self.g.add_node(node)
        self.assertTrue(node in self.g.user_nodes)
        self.g.remove_node(node)
        self.assertFalse(node in self.g.user_nodes)

        # remove multiple nodes
        nodes = [UserNode("to_remove"), ItemNode("to_remove")]
        self.g.add_node(nodes)
        self.assertTrue(nodes[0] in self.g.user_nodes)
        self.assertTrue(nodes[1] in self.g.item_nodes)
        self.g.remove_node(nodes)
        self.assertFalse(nodes[0] in self.g.user_nodes)
        self.assertFalse(nodes[1] in self.g.item_nodes)

        # remove not existent node
        user_nodes_before_removal = self.g.user_nodes
        self.g.remove_node(UserNode("not existent"))
        user_nodes_after_removal = self.g.user_nodes
        self.assertEqual(user_nodes_before_removal, user_nodes_after_removal)

    def test_copy(self):
        copy = self.g.copy()
        self.assertTrue(copy == self.g)

        copy.add_node(UserNode('prova'))

        self.assertFalse(copy == self.g)

    def test_serialize(self):

        self.g.serialize('./test_serialize', 'test_graph')

        with lzma.open(os.path.join('./test_serialize/test_graph.xz'), 'rb') as graph_file:
            graph = pickle.load(graph_file)

        self.assertEqual(self.g, graph)

        shutil.rmtree("test_serialize")

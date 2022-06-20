import pandas as pd
from unittest import TestCase
import os

from clayrs.content_analyzer import Ratings
from clayrs.recsys import TopKDegreeCentrality
from clayrs.recsys.graphs.feature_selection import TopKPageRank
from clayrs.recsys.graphs.feature_selection import feature_selector
from clayrs.recsys.graphs.nx_implementation.nx_full_graphs import NXFullGraph
from test import dir_test_files


class TestFeatureSelection(TestCase):

    def setUp(self) -> None:
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
        user_dir = os.path.join(dir_test_files, 'complex_contents', 'users_codified/')

        df = pd.DataFrame.from_dict({'user_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                     'item_id': ["tt0112281", "tt0113041", "tt0112453", "tt0112346",
                                                 "tt0112302", "tt0113101", "tt0113277", "tt0113228"],
                                     'score': [5, 3, 1, 5, 4, 2, 2, 4]})
        rat = Ratings.from_dataframe(df)

        # all properties from the dbpedia item repr extracted
        self.g: NXFullGraph = NXFullGraph(rat,
                                          user_contents_dir=user_dir,
                                          item_contents_dir=movies_dir,
                                          item_exo_properties={'dbpedia'},
                                          user_exo_properties={'local'}
                                          )

        self.users_target_nodes = list(self.g.user_nodes)
        self.items_target_nodes = list(self.g.item_nodes)

    def test_feature_selection_both(self):

        user_alg = TopKPageRank(k=1)
        item_alg = TopKDegreeCentrality(k=2)

        graph_fs = feature_selector(self.g, fs_algorithm_user=user_alg, fs_algorithm_item=item_alg,
                                    user_target_nodes=self.users_target_nodes,
                                    item_target_nodes=self.items_target_nodes)

        self.assertNotEqual(graph_fs, self.g)

        # we manually perform feature selection in order to get all labels that should not be present in the graph
        # after feature selection is performed
        labels_to_remove = user_alg.perform(self.g, self.users_target_nodes, mode='to_remove')
        labels_to_remove.extend(item_alg.perform(self.g, self.items_target_nodes, mode='to_remove'))

        # here we check that labels less important found at the previous step are not present in the graph
        # after feature selection
        for property_node in graph_fs.property_nodes:
            predecessors = graph_fs.get_predecessors(property_node)
            for pred in predecessors:
                link_label = graph_fs.get_link_data(pred, property_node).get('label')
                self.assertNotIn(link_label, labels_to_remove)

        # test inplace feature selection
        graph_fs = feature_selector(self.g, fs_algorithm_user=user_alg, fs_algorithm_item=item_alg,
                                    user_target_nodes=self.users_target_nodes,
                                    item_target_nodes=self.items_target_nodes, inplace=True)

        self.assertEqual(graph_fs, self.g)

    def test_feature_selection_only_users(self):

        user_alg = TopKDegreeCentrality(k=3)

        graph_fs = feature_selector(self.g, fs_algorithm_user=user_alg, user_target_nodes=self.users_target_nodes)

        # we manually perform feature selection in order to get all labels that should not be present in the graph
        # after feature selection is performed
        labels_to_remove = user_alg.perform(self.g, self.users_target_nodes, mode='to_remove')

        # here we check that labels less important found at the previous step are not present in the graph
        # after feature selection
        for property_node in graph_fs.property_nodes:
            predecessors = graph_fs.get_predecessors(property_node)
            for pred in predecessors:
                link_label = graph_fs.get_link_data(pred, property_node).get('label')
                self.assertNotIn(link_label, labels_to_remove)

    def test_feature_selection_only_items(self):

        item_alg = TopKPageRank(k=3)

        graph_fs = feature_selector(self.g, fs_algorithm_item=item_alg, item_target_nodes=self.items_target_nodes)

        # we manually perform feature selection in order to get all labels that should not be present in the graph
        # after feature selection is performed
        labels_to_remove = item_alg.perform(self.g, self.items_target_nodes, mode='to_remove')

        # here we check that labels less important found at the previous step are not present in the graph
        # after feature selection
        for property_node in graph_fs.property_nodes:
            predecessors = graph_fs.get_predecessors(property_node)
            for pred in predecessors:
                link_label = graph_fs.get_link_data(pred, property_node).get('label')
                self.assertNotIn(link_label, labels_to_remove)

    def test_special_cases_feature_selection(self):
        # algorithm failed to converge both for items and users
        graph_fs = feature_selector(self.g, fs_algorithm_user=TopKPageRank(max_iter=0),
                                    fs_algorithm_item=TopKPageRank(max_iter=0))

        # since both fs algorithm failed, the graph remained untouched
        self.assertEqual(graph_fs, self.g)

        # check that the original graph is returned if no fs algorithm is defined
        graph_fs = feature_selector(self.g, user_target_nodes=self.users_target_nodes,
                                    item_target_nodes=self.items_target_nodes)

        self.assertEqual(graph_fs, self.g)

        # check that when no target list is defined, all users nodes and all items nodes are used as targets
        graph_fs_no_target = feature_selector(self.g, fs_algorithm_user=TopKPageRank(k=2),
                                              fs_algorithm_item=TopKPageRank(k=2))

        graph_fs_w_target = feature_selector(self.g, fs_algorithm_user=TopKPageRank(k=2),
                                             fs_algorithm_item=TopKPageRank(k=2),
                                             user_target_nodes=self.users_target_nodes,
                                             item_target_nodes=self.items_target_nodes)

        self.assertEqual(graph_fs_no_target, graph_fs_w_target)

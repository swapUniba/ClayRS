import pandas as pd
from unittest import TestCase
import os

from clayrs.content_analyzer import Ratings
from clayrs.recsys import PropertyNode
from clayrs.recsys.graphs.feature_selection import TopKPageRank, TopKDegreeCentrality, \
    TopKEigenVectorCentrality
from clayrs.recsys.graphs.feature_selection import FeatureSelectionException
from clayrs.recsys.graphs import NXFullGraph
from test import dir_test_files


class TestFeatureSelectionAlgorithm(TestCase):

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
                                          user_exo_properties={'local': '1'}
                                          )
        # we use as target nodes all item nodes in the graph, meaning that we want want to remove
        # less important features of all the item nodes of the graph
        self.target_item_nodes = list(self.g.item_nodes)

        # here we save in a set all unique property labels such as 'starring', 'editing', etc.
        self.unique_prop_labels = set()
        for item_node in self.target_item_nodes:
            prop_nodes = [node for node in self.g.get_successors(item_node) if isinstance(node, PropertyNode)]
            for prop_n in prop_nodes:
                link_label = self.g.get_link_data(item_node, prop_n)['label']
                self.unique_prop_labels.add(link_label)

        # 'film director' is one of the label that appears less often, so, since the example graph is pretty simple,
        # the result of feature selection algorithms should not include it when the mode is 'to_remove'
        self.less_important_property_label = 'film director'

        # 'starring' is one of the label that appears more often, so, since the example graph is pretty simple,
        # the result of feature selection algorithms should include it when the mode is 'to_keep'
        self.most_important_property_label = 'starring'

    def test_perform_NXTopKPageRank(self):
        top_prop_labels_to_keep = 3

        result = TopKPageRank(k=top_prop_labels_to_keep).perform(self.g, self.target_item_nodes, mode='to_remove')

        # mode -> 'to_remove' will return a list of the property labels to remove
        n_nodes_to_remove = len(self.unique_prop_labels) - top_prop_labels_to_keep
        self.assertEqual(len(result), n_nodes_to_remove)
        self.assertTrue(self.less_important_property_label in result)
        self.assertTrue(self.most_important_property_label not in result)

        result = TopKPageRank(k=top_prop_labels_to_keep).perform(self.g, self.target_item_nodes, mode='to_keep')

        # mode -> 'to_keep' will return a list of the property labels to keep
        self.assertEqual(len(result), top_prop_labels_to_keep)
        self.assertTrue(self.less_important_property_label not in result)
        self.assertTrue(self.most_important_property_label in result)

    def test_perform_NXTopKDegreeCentrality(self):
        top_prop_labels_to_keep = 3
        result = TopKDegreeCentrality(k=top_prop_labels_to_keep).perform(self.g, self.target_item_nodes,
                                                                         mode='to_remove')

        # mode -> 'to_remove' will return a list of the property labels to remove
        n_nodes_to_remove = len(self.unique_prop_labels) - top_prop_labels_to_keep
        self.assertEqual(len(result), n_nodes_to_remove)
        self.assertTrue(self.less_important_property_label in result)
        self.assertTrue(self.most_important_property_label not in result)

        result = TopKDegreeCentrality(k=top_prop_labels_to_keep).perform(self.g, self.target_item_nodes,
                                                                         mode='to_keep')

        # mode -> 'to_keep' will return a list of the property labels to keep
        self.assertEqual(len(result), top_prop_labels_to_keep)
        self.assertTrue(self.less_important_property_label not in result)
        self.assertTrue(self.most_important_property_label in result)

    def test_perform_NXTopKEigenVectorCentrality(self):
        top_prop_labels_to_keep = 3
        result = TopKEigenVectorCentrality(k=top_prop_labels_to_keep, max_iter=200).perform(self.g,
                                                                                            self.target_item_nodes,
                                                                                            mode='to_remove')

        # mode -> 'to_remove' will return a list of the property labels to remove
        n_nodes_to_remove = len(self.unique_prop_labels) - top_prop_labels_to_keep
        self.assertEqual(len(result), n_nodes_to_remove)
        self.assertTrue(self.less_important_property_label in result)
        self.assertTrue(self.most_important_property_label not in result)

        result = TopKEigenVectorCentrality(k=top_prop_labels_to_keep, max_iter=200).perform(self.g,
                                                                                            self.target_item_nodes,
                                                                                            mode='to_keep')

        # mode -> 'to_keep' will return a list of the property labels to keep
        self.assertEqual(len(result), top_prop_labels_to_keep)
        self.assertTrue(self.less_important_property_label not in result)
        self.assertTrue(self.most_important_property_label in result)

    def test_special_cases_feature_selection(self):
        # empty target nodes
        self.assertEqual(TopKPageRank(k=2).perform(self.g, []), [])

        # k = 0
        result = TopKPageRank(k=0).perform(self.g, self.target_item_nodes)
        self.assertEqual(result, [])

        # k < 0
        with self.assertRaises(ValueError):
            TopKPageRank(k=-1)

        # k > number of properties in the graph
        result = TopKPageRank(100).perform(self.g, self.target_item_nodes)
        self.assertEqual(len(result), len(self.unique_prop_labels))  # number of item labels

        # method fails to converge
        with self.assertRaises(FeatureSelectionException):
            TopKPageRank(k=1, max_iter=0).perform(self.g, self.target_item_nodes)

        # non existent mode
        with self.assertRaises(TypeError):
            TopKPageRank().perform(self.g, [], mode='invalid')

import pandas as pd
from unittest import TestCase
import os

from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection import NXTopKPageRank
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection import FeatureSelectionException
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection_handler import \
    FeatureSelectionHandler
from orange_cb_recsys.recsys.graphs.graph import FullGraph
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
from orange_cb_recsys.utils.const import root_path


class TestFeatureSelectionHandler(TestCase):

    def assertPropNumber(self, result: FullGraph, user_or_item_nodes: set, expected_prop_number: int):
        # used to check that the expected number of properties matches the number of properties in the graph
        # returned by the FeatureSelectionHandler

        actual_prop = set()

        for node in user_or_item_nodes:
            for successor in result.get_successors(node):
                if result.is_property_node(successor):
                    property_label = result.get_link_data(node, successor)['label']
                    actual_prop.add(property_label)

        if expected_prop_number != len(actual_prop):
            raise AssertionError("Expected %s properties but %s found" % (expected_prop_number, len(actual_prop)))

    def setUp(self) -> None:
        contents_path = os.path.join(root_path, 'contents')
        movies_dir = os.path.join(contents_path, 'movies_codified/')
        user_dir = os.path.join(contents_path, 'users_codified/')

        self.df = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                          'to_id': ["tt0113228", "tt0113041", "tt0113228", "tt0112346",
                                                    "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                                          'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        # all properties from the dbpedia item repr extracted
        self.g_None_item_prop: NXFullGraph = NXFullGraph(self.df,
                                                         user_contents_dir=user_dir,
                                                         item_contents_dir=movies_dir,
                                                         item_exo_representation='dbpedia',
                                                         user_exo_representation='local',
                                                         item_exo_properties=None,
                                                         user_exo_properties=['1']
                                                         )

        # all representations for the defined item properties are extracted
        self.g_None_item_repr: NXFullGraph = NXFullGraph(self.df,
                                                         user_contents_dir=user_dir,
                                                         item_contents_dir=movies_dir,
                                                         item_exo_representation=None,
                                                         user_exo_representation='local',
                                                         item_exo_properties=['starring', 'editing', 'producer',
                                                                              'writer'],
                                                         user_exo_properties=['1']
                                                         )

        user_4_items = list(set(self.df.query("from_id == '4'")['to_id']))
        recommendable_items_for_user_4 = list(set(self.df.query("to_id not in @user_4_items")['to_id']))
        users = list(set(self.df['from_id']))

        self.target_user_nodes = users
        self.target_item_nodes = recommendable_items_for_user_4

    def test_feature_selection_handler(self):
        result = FeatureSelectionHandler(NXTopKPageRank(2)).\
            process_feature_selection_on_fullgraph(self.g_None_item_prop, self.target_user_nodes,
                                                   self.target_item_nodes)

        self.assertEqual(self.g_None_item_prop.item_nodes, result.item_nodes)
        self.assertEqual(self.g_None_item_prop.user_nodes, result.user_nodes)
        self.assertLess(len(result.property_nodes), len(self.g_None_item_prop.property_nodes))
        # only 1 because 1 user property is defined for the graph
        self.assertPropNumber(result, self.g_None_item_prop.user_nodes, 1)
        self.assertPropNumber(result, self.g_None_item_prop.item_nodes, 2)

        result = FeatureSelectionHandler(NXTopKPageRank(2)). \
            process_feature_selection_on_fullgraph(self.g_None_item_repr, self.target_user_nodes,
                                                   self.target_item_nodes)

        self.assertEqual(self.g_None_item_repr.item_nodes, result.item_nodes)
        self.assertEqual(self.g_None_item_repr.user_nodes, result.user_nodes)
        self.assertLess(len(result.property_nodes), len(self.g_None_item_repr.property_nodes))
        self.assertPropNumber(result, self.g_None_item_repr.user_nodes, 1)
        self.assertPropNumber(result, self.g_None_item_repr.item_nodes, 2)

    def test_get_property_labels_info(self):
        labels = FeatureSelectionHandler(NXTopKPageRank()).\
            _get_property_labels_info(self.g_None_item_repr, set(self.df['to_id']))

        self.assertEqual(set(labels), set(self.g_None_item_repr.get_item_exogenous_properties()))

    def test_special_cases_feature_selection_handler(self):
        # algorithm failed to converge both for items and users
        result = FeatureSelectionHandler(NXTopKPageRank(2, max_iter=0)).\
            process_feature_selection_on_fullgraph(self.g_None_item_prop, self.target_user_nodes,
                                                   self.target_item_nodes)
        self.assertTrue(result is self.g_None_item_prop)

        # target user nodes list badly defined
        with self.assertRaises(FeatureSelectionException):
            result = FeatureSelectionHandler(NXTopKPageRank(2)). \
                process_feature_selection_on_fullgraph(self.g_None_item_prop, ['not_an_user_node'],
                                                       self.target_item_nodes)

        # target item nodes list badly defined
        with self.assertRaises(FeatureSelectionException):
            result = FeatureSelectionHandler(NXTopKPageRank(2)). \
                process_feature_selection_on_fullgraph(self.g_None_item_prop, self.target_user_nodes,
                                                       ['not_an_item_node'])

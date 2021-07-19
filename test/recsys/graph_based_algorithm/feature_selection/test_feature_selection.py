import pandas as pd
from unittest import TestCase
import os

from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection import NXTopKPageRank, NXTopKDegreeCentrality, \
    NXTopKEigenVectorCentrality
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.exceptions import FeatureSelectionException
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
from orange_cb_recsys.utils.const import root_path


class TestFeatureSelection(TestCase):

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

        self.target_item_nodes = recommendable_items_for_user_4

        # producer is the label that appears less often, so, since the example graph is pretty simple,
        # the result of feature selection algorithms should not include it
        self.less_important_property_label_None_prop = 'producer'

        # same as above
        self.less_important_property_label_None_repr = 'producer#0#dbpedia'

    def test_perform_NXTopKPageRank(self):
        result = NXTopKPageRank(2).perform(self.g_None_item_prop, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_prop not in result)

        result = NXTopKPageRank(2).perform(self.g_None_item_repr, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_repr not in result)

    def test_perform_NXTopKDegreeCentrality(self):
        result = NXTopKDegreeCentrality(2).perform(self.g_None_item_prop, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_prop not in result)

        result = NXTopKDegreeCentrality(2).perform(self.g_None_item_repr, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_repr not in result)

    def test_perform_NXTopKEigenVectorCentrality(self):
        result = NXTopKEigenVectorCentrality(2).perform(self.g_None_item_prop, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_prop not in result)

        result = NXTopKEigenVectorCentrality(2).perform(self.g_None_item_repr, self.target_item_nodes)
        self.assertEqual(len(result), 2)
        self.assertTrue(self.less_important_property_label_None_repr not in result)

    def test_special_cases_feature_selection(self):
        # k = 0
        self.assertEqual(NXTopKPageRank(0).perform_feature_selection(self.g_None_item_prop, []), {})

        # k > number of properties in the graph
        result = NXTopKPageRank(100).perform(self.g_None_item_repr, self.target_item_nodes)
        self.assertEqual(len(result), 4)  # number of item properties

        # method fails to converge
        with self.assertRaises(FeatureSelectionException):
            NXTopKPageRank(1, max_iter=0).perform(self.g_None_item_repr, self.target_item_nodes)

        # no target nodes defined
        with self.assertRaises(FeatureSelectionException):
            NXTopKPageRank().perform(self.g_None_item_prop, [])

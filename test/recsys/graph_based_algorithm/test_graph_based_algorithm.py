from unittest import TestCase
import pandas as pd

from clayrs.content_analyzer import Ratings
from clayrs.recsys.graphs.graph import ItemNode, UserNode, PropertyNode
from clayrs.recsys.graphs.nx_implementation.nx_full_graphs import NXFullGraph

from clayrs.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank


class TestGraphBasedAlgorithm(TestCase):

    def setUp(self) -> None:
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", -0.2, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0113041", 0.6, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        self.graph = NXFullGraph(ratings)

        # GraphBasedAlgorithm is an abstract class, so we instantiate a subclass in order to test its methods
        self.alg = NXPageRank()

    def test_filter_result(self):
        rank = {UserNode("A000"): 0.5, ItemNode("tt0114576"): 0.5, UserNode("A001"): 0.5, ItemNode("tt0113497"): 0.5,
                ItemNode("tt0112453"): 0.5, PropertyNode("Nolan"): 0.5}

        # filter list with item i1, in this case graph parameter and user node parameter won't do anything
        result = self.alg.filter_result(graph=self.graph, result=rank, filter_list=[ItemNode('tt0114576')],
                                        user_node=UserNode("A000"))
        expected = {ItemNode("tt0114576"): 0.5}
        self.assertEqual(expected, result)

        # filter list with item i1 and item i2, in this case graph parameter and user node parameter won't do anything
        result = self.alg.filter_result(graph=self.graph, result=rank, filter_list=[ItemNode('tt0114576'),
                                                                                    PropertyNode('Nolan')],
                                        user_node=UserNode("A000"))
        expected = {ItemNode('tt0114576'): 0.5, PropertyNode("Nolan"): 0.5}
        self.assertEqual(expected, result)

        # filter with non existent nodes, result will be empty
        # in this case graph parameter and user node parameter won't do anything
        result = self.alg.filter_result(graph=self.graph, result=rank, filter_list=[ItemNode('non_existent')],
                                        user_node=UserNode("A000"))
        expected = {}
        self.assertEqual(expected, result)

        # clean result for user A000, the cleaned result will have only item nodes
        result = self.alg.filter_result(graph=self.graph, result=rank, filter_list=None,
                                        user_node=UserNode("A000"))
        expected = {ItemNode("tt0113497"): 0.5}

        self.assertEqual(expected, result)

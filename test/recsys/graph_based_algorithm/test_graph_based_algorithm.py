from unittest import TestCase
import pandas as pd

from orange_cb_recsys.recsys.graphs.graph import ItemNode, UserNode
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank


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

        # ContentBasedAlgorithm is an abstract class, so we need to instantiate
        # a subclass to test its methods
        self.graph = NXFullGraph(ratings)

        self.graph.add_property_node('Nolan')

        self.alg = NXPageRank()

    def test_clean_rank(self):
        rank = {"A000": 0.5, "tt0114576": 0.5, "A001": 0.5, "tt0113497": 0.5, "tt0112453": 0.5, "Nolan": 0.5}

        # remove from rank all nodes except Item nodes
        result = self.alg.clean_result(self.graph, rank, user_id="A000")
        expected = {"tt0113497": 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and User nodes
        result = self.alg.clean_result(self.graph, rank, user_id="A000", remove_users=False)
        expected = {"tt0113497": 0.5, "A001": 0.5, "A000": 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and keep item rated by the user
        result = self.alg.clean_result(self.graph, rank, user_id="A000", remove_profile=False)
        expected = {'tt0112453': 0.5, 'tt0113497': 0.5, 'tt0114576': 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and property nodes
        result = self.alg.clean_result(self.graph, rank, user_id="A000", remove_properties=False)
        expected = {'tt0113497': 0.5, 'Nolan': 0.5}
        self.assertEqual(expected, result)

    def test_extract_profile(self):

        result = self.alg.extract_profile(self.graph, "A000")
        expected = {'tt0112453': -0.2, 'tt0113041': 0.6, 'tt0114576': 1.0}

        self.assertEqual(expected, result)

        # Also if you wrap items in its corresponding type will work
        expected_wrapped = {ItemNode('tt0112453'): -0.2, ItemNode('tt0113041'): 0.6, ItemNode('tt0114576'): 1.0}
        self.assertEqual(expected_wrapped, result)

        # This will fail because they are not users
        expected_wrapped_fake = {UserNode('tt0112453'): -0.2, UserNode('tt0113041'): 0.6, UserNode('tt0114576'): 1.0}
        self.assertNotEqual(expected_wrapped_fake, result)

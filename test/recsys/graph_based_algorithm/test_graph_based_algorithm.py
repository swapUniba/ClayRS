from unittest import TestCase
import pandas as pd
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank import NXPageRank


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
        graph = NXFullGraph(ratings)

        graph.add_property_node('Nolan')

        self.alg = NXPageRank()
        self.alg.initialize(graph)

    def test_clean_rank(self):
        rank = {"A000": 0.5, "tt0114576": 0.5, "A001": 0.5, "tt0113497": 0.5, "tt0112453": 0.5, "Nolan": 0.5}

        # remove from rank all nodes except Item nodes
        result = self.alg.clean_result(rank, user_id="A000")
        expected = {"tt0113497": 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and User nodes
        result = self.alg.clean_result(rank, user_id="A000", remove_users=False)
        expected = {"tt0113497": 0.5, "A001": 0.5, "A000": 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and keep item rated by the user
        result = self.alg.clean_result(rank, user_id="A000", remove_profile=False)
        expected = {'tt0112453': 0.5, 'tt0113497': 0.5, 'tt0114576': 0.5}
        self.assertEqual(expected, result)

        # remove from rank all nodes except Item nodes and property nodes
        result = self.alg.clean_result(rank, user_id="A000", remove_properties=False)
        expected = {'tt0113497': 0.5, 'Nolan': 0.5}
        self.assertEqual(expected, result)

    def test_extract_profile(self):

        result = self.alg.extract_profile("A000")
        expected = {'tt0112453': 0.4, 'tt0113041': 0.8, 'tt0114576': 1.0}

        self.assertEqual(expected, result)

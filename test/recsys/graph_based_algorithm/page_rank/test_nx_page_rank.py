import pandas as pd
from unittest import TestCase

from clayrs.content_analyzer import Ratings
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from clayrs.recsys.graphs import NXFullGraph

ratings = pd.DataFrame.from_records([
    ("A000", "tt0114576", 1, "54654675"),
    ("A000", "tt0112453", 0.5, "54654675"),
    ("A001", "tt0114576", 0.5, "54654675"),
    ("A001", "tt0112896", 0, "54654675"),
    ("A000", "tt0113041", 0.75, "54654675"),
    ("A002", "tt0112453", 0.5, "54654675"),
    ("A002", "tt0113497", 0.5, "54654675"),
    ("A003", "tt0112453", 0, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])
ratings = Ratings.from_dataframe(ratings, timestamp_column="timestamp")

test_set = pd.DataFrame.from_records([
    ("A000", "tt0113497", 1, "54654675"),
    ("A001", "tt0112453", 0.5, "54654675"),
    ("A002", "tt0114576", 0.5, "54654675"),
    ("A003", "tt0114576", 0, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])
test_set = Ratings.from_dataframe(test_set, timestamp_column="timestamp")


class TestNXPageRank(TestCase):

    def setUp(self) -> None:
        self.graph = NXFullGraph(ratings)

    def test_predict(self):
        alg = NXPageRank()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict({'A000'}, self.graph, test_set)

    def test_rank(self):
        # test not personalized
        alg = NXPageRank()

        # rank with None methodology (all unrated items will be ranked)
        res_all_unrated = alg.rank({'A000'}, self.graph, test_set, methodology=None, num_cpus=1)
        item_rated_set = set([interaction.item_id for interaction in ratings.get_user_interactions("A000")])
        item_ranked_set = set([pred_interaction.item_id for pred_interaction in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 1
        res_n_recs = alg.rank({'A000'}, self.graph, test_set, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set([interaction.item_id for interaction in ratings.get_user_interactions("A000")])
        item_ranked_set = set([pred_interaction.item_id for pred_interaction in res_n_recs])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # test personalized
        alg = NXPageRank(personalized=True)
        result_personalized = alg.rank({'A000'}, self.graph, test_set, num_cpus=1)

        alg = NXPageRank()
        result_not_personalized = alg.rank({'A000'}, self.graph, test_set, num_cpus=1)

        self.assertNotEqual(result_personalized, result_not_personalized)

        # test with custom parameters
        alg = NXPageRank(alpha=0.9, max_iter=4, tol=1e-3)
        result_custom = alg.rank({'A000'}, self.graph, test_set, num_cpus=1)

        alg = NXPageRank()
        result_not_custom = alg.rank({'A000'}, self.graph, test_set, num_cpus=1)

        self.assertNotEqual(result_custom, result_not_custom)

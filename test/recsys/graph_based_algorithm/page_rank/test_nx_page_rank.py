import pandas as pd
from unittest import TestCase
import numpy as np

from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
from orange_cb_recsys.utils.feature_selection import NXFSPageRank


class TestNXPageRank(TestCase):

    def setUp(self) -> None:
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", 0.5, "54654675"),
            ("A001", "tt0114576", 0.5, "54654675"),
            ("A001", "tt0112896", 0, "54654675"),
            ("A000", "tt0113041", 0.75, "54654675"),
            ("A002", "tt0112453", 0.5, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", 0, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        self.graph = NXFullGraph(self.ratings)

        self.filter_list = ['tt0114576', 'tt0112453', 'tt0113497']

    def test_predict(self):
        alg = NXPageRank()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict('A000', self.graph)

    def test_rank(self):

        # test not personalized
        alg = NXPageRank()

        # rank with filter_list
        res_filtered = alg.rank('A000', self.graph, filter_list=self.filter_list)
        item_ranked_set = set(res_filtered['to_id'])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank('A000', self.graph)
        item_rated_set = set(self.ratings.query('from_id == "A000"')['to_id'])
        item_ranked_set = set(res_all_unrated['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 1
        res_n_recs = alg.rank('A000', self.graph, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set(self.ratings.query('from_id == "A000"')['to_id'])
        item_ranked_set = set(res_n_recs['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # test personalized
        alg = NXPageRank(personalized=True)
        result_personalized = alg.rank('A000', self.graph)

        alg = NXPageRank()
        result_not_personalized = alg.rank('A000', self.graph)

        result_personalized = np.array(result_personalized)
        result_not_personalized = np.array(result_not_personalized)

        result_personalized.sort(axis=0)
        result_not_personalized.sort(axis=0)

        self.assertFalse(np.array_equal(result_personalized, result_not_personalized))

        # alg = NXPageRank(graph=graph)
        # rank_fs = alg.predict('A001', ratings, 1, feature_selection_algorithm=NXFSPageRank())
        # logger.info('pg_rk results')
        # for r in rank_fs.keys():
        #     print(str(r) + " " + str(rank_fs[r]))

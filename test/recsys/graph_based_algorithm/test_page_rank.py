import pandas as pd
from unittest import TestCase

from orange_cb_recsys.recsys.graph_based_algorithm.page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.utils.feature_selection import NXFSPageRank


class TestNXPageRank(TestCase):

    def setUp(self) -> None:
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", -0.2, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0113041", 0.6, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        self.graph = NXFullGraph(self.ratings)

        self.filter_list = ['tt0114576', 'tt0112453', 'tt0113497']

    def test_predict(self):
        alg = NXPageRank()
        alg.initialize(self.graph)

        pred_result = alg.fit_prediction('A000', self.filter_list)
        self.assertEqual(len(pred_result), len(self.filter_list))

        rank_result = alg.fit_rank('A000', recs_number=2)
        self.assertEqual(len(rank_result), 2)

        alg = NXPageRank(personalized=True)
        alg.initialize(self.graph)

        pred_result = alg.fit_prediction('A000', self.filter_list)
        self.assertEqual(len(pred_result), len(self.filter_list))

        rank_result = alg.fit_rank('A000', recs_number=2)
        self.assertEqual(len(rank_result), 2)

        # alg = NXPageRank(graph=graph)
        # rank_fs = alg.predict('A001', ratings, 1, feature_selection_algorithm=NXFSPageRank())
        # logger.info('pg_rk results')
        # for r in rank_fs.keys():
        #     print(str(r) + " " + str(rank_fs[r]))

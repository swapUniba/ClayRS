import pandas as pd
from unittest import TestCase

from orange_cb_recsys.recsys import NXPageRank
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.feature_selection import NXFSPageRank

ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0113041", 0.6, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

graph = NXFullGraph(ratings)


class TestNXPageRank(TestCase):
    def test_predict(self):
        alg = NXPageRank(graph=graph)
        rank = alg.predict('A001', ratings, 1)
        logger.info('pg_rk results')
        for r in rank.keys():
            print(str(r) + " " + str(rank[r]))

        self.assertIn('tt0112453', rank.keys())

        # alg = NXPageRank(graph=graph)
        # rank_fs = alg.predict('A001', ratings, 1, feature_selection_algorithm=NXFSPageRank())
        # logger.info('pg_rk results')
        # for r in rank_fs.keys():
        #     print(str(r) + " " + str(rank_fs[r]))

        alg = NXPageRank(graph=graph, personalized=True)
        rank_personalized = alg.predict('A001', ratings, 1)
        logger.info('pg_rk results')
        for r in rank_personalized.keys():
            print(str(r) + " " + str(rank_personalized[r]))

        self.assertIn('tt0113041', rank_personalized)


class PageRankAlg(TestCase):
    def test_clean_rank(self):
        rank = {"A000": 0.5, "tt0114576": 0.5, "A001": 0.5, "tt0113497": 0.5, "tt0112453": 0.5}
        alg = NXPageRank(graph=graph)

        # remove from rank all from nodes
        result = alg.clean_rank(rank, user_id="A000", remove_profile=False, remove_from_nodes=True)
        expected = {"tt0114576": 0.5, "tt0113497": 0.5, "tt0112453": 0.5}
        self.assertEqual(expected, result)

        # remove from rank all from nodes and all data from user A000
        result = alg.clean_rank(rank, user_id="A000", remove_profile=True, remove_from_nodes=True)
        expected = {"tt0113497": 0.5}
        self.assertEqual(expected, result)

    def test_extract_profile(self):
        alg = NXPageRank(graph=graph)
        result = alg.extract_profile("A000")

        expected = {'tt0114576': 0.75, 'tt0112453': 0.25, 'tt0113041': 0.8}

        self.assertEqual(expected, result)
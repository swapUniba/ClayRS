import pandas as pd
from unittest import TestCase
import numpy as np
import os

from orange_cb_recsys.recsys import NXTopKPageRank
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs.nx_full_graphs import NXFullGraph
from orange_cb_recsys.utils.const import root_path


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

    def test_page_rank_with_feature_selection(self):
        # the PageRank algorithm is tested with the NXTopKPageRank Feature Selection algorithm
        # since the Feature Selection is already tested in the dedicated test file
        # this test only checks that the PageRank run works while defining a Feature Selection algorithm

        contents_path = os.path.join(root_path, 'contents')
        movies_dir = os.path.join(contents_path, 'movies_codified/')
        user_dir = os.path.join(contents_path, 'users_codified/')

        df = pd.DataFrame.from_dict({'from_id': ["1", "1", "2", "2", "2", "3", "4", "4"],
                                     'to_id': ["tt0113228", "tt0113041", "tt0113228", "tt0112346",
                                               "tt0112453", "tt0112453", "tt0112346", "tt0112453"],
                                     'score': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        # only one property from the dbpedia repr extracted
        graph_with_properties: NXFullGraph = NXFullGraph(df,
                                                         user_contents_dir=user_dir,
                                                         item_contents_dir=movies_dir,
                                                         item_exo_representation='dbpedia',
                                                         user_exo_representation='local',
                                                         item_exo_properties=None,
                                                         user_exo_properties=['1']
                                                         )

        # fs standard algorithm
        alg = NXPageRank(feature_selection=NXTopKPageRank())
        result = alg.rank('4', graph_with_properties)
        self.assertEqual(len(result), 2)

        # fs personalized algorithm
        alg = NXPageRank(personalized=True, feature_selection=NXTopKPageRank())
        result_personalized = alg.rank('4', graph_with_properties)
        self.assertEqual(len(result_personalized), 2)

        # fs personalized algorithm and filter list
        alg = NXPageRank(personalized=True, feature_selection=NXTopKPageRank())
        result_personalized = alg.rank('4', graph_with_properties, filter_list=['tt0113228'])
        self.assertEqual(len(result_personalized), 1)

        # fs personalized algorithm and empty filter list
        alg = NXPageRank(personalized=True, feature_selection=NXTopKPageRank())
        result_personalized = alg.rank('4', graph_with_properties, filter_list=[])
        self.assertEqual(len(result_personalized), 0)


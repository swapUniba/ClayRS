import os
from unittest import TestCase
import pandas as pd
from orange_cb_recsys.recsys import RecSys, ContentBasedConfig, GraphBasedConfig
from orange_cb_recsys.recsys.content_based_algorithm import ClassifierRecommender, CentroidVector, CosineSimilarity, \
    IndexQuery
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs import NXFullGraph

from orange_cb_recsys.utils.const import root_path

from orange_cb_recsys.recsys.content_based_algorithm.classifier import SVM

contents_path = os.path.join(root_path, 'contents')

class TestRecSys(TestCase):

    def setUp(self) -> None:
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", 0.1, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0113041", -1, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])


    def test_content_based_recsys(self):
        movies_multiple = os.path.join(contents_path, 'movies_multiple_repr/')
        filter_list = ['tt0114319', 'tt0114388']
        recs_number = 3

        self.centroid_vector(movies_multiple, filter_list, recs_number)
        self.classifier_recommender(movies_multiple, filter_list, recs_number)

        movies_index = os.path.join(contents_path, 'movies_multiple_repr_INDEX/')
        self.index_query(movies_index, filter_list, recs_number)

    def centroid_vector(self, movies_multiple: str, filter_list: list, recs_number: int):
        # Test prediction and ranking with the Centroid Vector algorithm
        alg = CentroidVector({'Plot': ['0', '1']}, CosineSimilarity(), threshold=0)
        config = ContentBasedConfig(alg, self.ratings, movies_multiple)

        result_prediction = RecSys(config).fit_predict('A000')
        result_rank = RecSys(config).fit_ranking('A000')

        self.assertEqual(len(result_prediction), len(result_rank))

        # Test prediction and ranking with the Centroid Vector algorithm on specified items
        result_prediction_filtered = RecSys(config).fit_predict('A000', filter_list=filter_list)
        result_rank_filtered = RecSys(config).fit_ranking('A000', filter_list=filter_list)

        self.assertEqual(len(result_prediction_filtered), len(filter_list))
        self.assertEqual(len(result_rank_filtered), len(filter_list))

        # Test top-n ranking with the Centroid Vector algorithm
        result_rank_numbered = RecSys(config).fit_ranking('A000', recs_number=recs_number)

        self.assertEqual(len(result_rank_numbered), recs_number)

    def classifier_recommender(self, movies_multiple: str, filter_list: list, recs_number: int):
        # Test prediction and ranking with the Classifier Recommender algorithm
        alg = ClassifierRecommender({'Plot': ['0', '1']}, SVM(), threshold=0)
        config = ContentBasedConfig(alg, self.ratings, movies_multiple)

        result_prediction = RecSys(config).fit_predict('A000')
        result_rank = RecSys(config).fit_ranking('A000')

        self.assertEqual(len(result_prediction), len(result_rank))

        # Test prediction and ranking with the Classifier Recommender algorithm on specified items
        result_prediction_filtered = RecSys(config).fit_predict('A000', filter_list=filter_list)
        result_rank_filtered = RecSys(config).fit_ranking('A000', filter_list=filter_list)

        self.assertEqual(len(result_prediction_filtered), len(filter_list))
        self.assertEqual(len(result_rank_filtered), len(filter_list))

        # Test top-n ranking with the Classifier Recommender algorithm
        result_rank_numbered = RecSys(config).fit_ranking('A000', recs_number=recs_number)

        self.assertEqual(len(result_rank_numbered), recs_number)

    def index_query(self, movies_index: str, filter_list: list, recs_number: int):
        # Test prediction and ranking with the Index Query algorithm
        alg = IndexQuery({'Plot': ['0', '1']}, threshold=0)
        config = ContentBasedConfig(alg, self.ratings, movies_index)

        result_prediction = RecSys(config).fit_predict('A000')
        result_rank = RecSys(config).fit_ranking('A000')

        self.assertEqual(len(result_prediction), len(result_rank))

        # Test prediction and ranking with the IndexQuery algorithm on specified items
        result_prediction_filtered = RecSys(config).fit_predict('A000', filter_list=filter_list)
        result_rank_filtered = RecSys(config).fit_ranking('A000', filter_list=filter_list)

        self.assertEqual(len(result_prediction_filtered), len(filter_list))
        self.assertEqual(len(result_rank_filtered), len(filter_list))

        # Test top-n ranking with the IndexQuery algorithm
        result_rank_numbered = RecSys(config).fit_ranking('A000', recs_number=recs_number)

        self.assertEqual(len(result_rank_numbered), recs_number)

    def test_graph_based_recsys(self):
        # Because graph based recommendation needs to have all items in the ratings dataframe
        filter_list = ['tt0112896', 'tt0113497']
        recs_number = 2

        self.nx_page_rank(filter_list, recs_number)

    def nx_page_rank(self, filter_list: list, recs_number: int):
        graph = NXFullGraph(self.ratings)
        alg = NXPageRank()
        config = GraphBasedConfig(alg, graph)

        # Test prediction and ranking with the Page Rank algorithm
        result_prediction = RecSys(config).fit_predict('A000')
        result_rank = RecSys(config).fit_ranking('A000')

        self.assertEqual(len(result_prediction), len(result_rank))

        # Test prediction and ranking with the Centroid Vector algorithm on specified items
        result_prediction_filtered = RecSys(config).fit_predict('A000', filter_list=filter_list)
        result_rank_filtered = RecSys(config).fit_ranking('A000', filter_list=filter_list)

        self.assertEqual(len(result_prediction_filtered), len(filter_list))
        self.assertEqual(len(result_rank_filtered), len(filter_list))

        # Test top-n ranking with the Centroid Vector algorithm
        result_rank_numbered = RecSys(config).fit_ranking('A000', recs_number=recs_number)

        self.assertEqual(len(result_rank_numbered), recs_number)
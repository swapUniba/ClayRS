import os
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import SkSVC
from orange_cb_recsys.recsys.recsys import GraphBasedRS, ContentBasedRS
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifier_recommender import ClassifierRecommender
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
from orange_cb_recsys.recsys.content_based_algorithm.index_query.index_query import IndexQuery
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs import NXFullGraph

from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')

ratings = pd.DataFrame.from_records([
    ("A000", "tt0114576", 5, "54654675"),
    ("A001", "tt0114576", 3, "54654675"),
    ("A001", "tt0112896", 1, "54654675"),
    ("A000", "tt0113041", 1, "54654675"),
    ("A002", "tt0112453", 2, "54654675"),
    ("A002", "tt0113497", 4, "54654675"),
    ("A003", "tt0112453", 1, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])


class TestContentBasedRS(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.movies_multiple = os.path.join(contents_path, 'movies_codified/')
        cls.filter_list = ['tt0114319', 'tt0114388']

    def test_empty_frame(self):
        ratings_only_positive = pd.DataFrame.from_records([
            ("A000", "tt0114576", 5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        ratings_only_negative = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        ratings_item_inexistent = pd.DataFrame.from_records([
            ("A000", "not exists", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        # ClassifierRecommender returns an empty frame
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC(), threshold=3)
        rs = ContentBasedRS(alg, ratings_only_positive, self.movies_multiple)
        result = rs.fit_rank('A000')
        self.assertTrue(result.empty)

        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC(), threshold=3)
        rs = ContentBasedRS(alg, ratings_only_negative, self.movies_multiple)
        result = rs.fit_rank('A000')
        self.assertTrue(result.empty)

        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC(), threshold=3)
        rs = ContentBasedRS(alg, ratings_item_inexistent, self.movies_multiple)
        result = rs.fit_rank('A000')
        self.assertTrue(result.empty)

        # CentroidVector returns an empty frame
        alg = CentroidVector({'Plot': ['tfidf', 'embedding']}, CosineSimilarity(), threshold=3)
        rs = ContentBasedRS(alg, ratings_only_negative, self.movies_multiple)
        result = rs.fit_rank('A000')
        self.assertTrue(result.empty)

        alg = CentroidVector({'Plot': ['tfidf', 'embedding']}, CosineSimilarity(), threshold=3)
        rs = ContentBasedRS(alg, ratings_item_inexistent, self.movies_multiple)
        result = rs.fit_rank('A000')
        self.assertTrue(result.empty)

    # More tests in content_based_algorithm/test_classifier
    def test_classifier_recommender(self):
        recs_number = 3

        # Test prediction and ranking with the Classifier Recommender algorithm
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC())
        rs = ContentBasedRS(alg, ratings, self.movies_multiple)

        # Prediction should raise error since it's not a ScorePredictionAlg
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000')

        # Test ranking with the Classifier Recommender algorithm on specified items
        result_rank_filtered = rs.fit_rank('A000', filter_list=self.filter_list)
        self.assertEqual(len(result_rank_filtered), len(self.filter_list))

        # Test top-n ranking with the Classifier Recommender algorithm
        result_rank_numbered = rs.fit_rank('A000', recs_number=recs_number)
        self.assertEqual(len(result_rank_numbered), recs_number)

    def test_centroid_vector(self):
        recs_number = 3

        # Test prediction and ranking with the Centroid Vector algorithm
        alg = CentroidVector({'Plot': ['tfidf', 'embedding']}, CosineSimilarity())
        rs = ContentBasedRS(alg, ratings, self.movies_multiple)

        # Prediction should raise error since it's not a ScorePredictionAlg
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000')

        # Test ranking with the Centroid Vector algorithm on specified items
        result_rank_filtered = rs.fit_rank('A000', filter_list=self.filter_list)
        self.assertEqual(len(result_rank_filtered), len(self.filter_list))

        # Test top-n ranking with the Centroid Vector algorithm
        result_rank_numbered = rs.fit_rank('A000', recs_number=recs_number)
        self.assertEqual(len(result_rank_numbered), recs_number)

    def test_index_query(self):
        movies_index = os.path.join(contents_path, 'index/')
        filter_list = ['tt0114319', 'tt0114388']
        recs_number = 3

        # Test prediction and ranking with the Index Query algorithm
        alg = IndexQuery({'Plot': ['index_original', 'index_preprocessed']})
        rs = ContentBasedRS(alg, ratings, movies_index)

        # Prediction should raise error since it's not a ScorePredictionAlg
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000')

        result_rank = rs.fit_rank('A000')
        self.assertGreater(len(result_rank), 0)

        # Test prediction and ranking with the IndexQuery algorithm on specified items, prediction will raise exception
        # since it's not a PredictionAlgorithm
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000', filter_list=filter_list)

        result_rank_filtered = rs.fit_rank('A000', filter_list=filter_list)
        self.assertGreater(len(result_rank_filtered), 0)

        # Test top-n ranking with the IndexQuery algorithm
        result_rank_numbered = rs.fit_rank('A000', recs_number=recs_number)
        self.assertEqual(len(result_rank_numbered), recs_number)


class TestGraphBasedRS(TestCase):
    def test_nx_page_rank(self):
        # Because graph based recommendation needs to have all items to predict in the ratings dataframe
        filter_list = ['tt0112896', 'tt0113497']
        recs_number = 1

        graph = NXFullGraph(ratings)
        alg = NXPageRank()
        rs = GraphBasedRS(alg, graph)

        # Test prediction and ranking with the Page Rank algorithm, prediction will raise exception
        # since it's not a PredictionAlgorithm
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000')

        result_rank = rs.fit_rank('A000')
        self.assertEqual(len(result_rank), 3)

        # Test prediction and ranking with the Page Rank algorithm on specified items, prediction will raise exception
        # since it's not a PredictionAlgorithm
        with self.assertRaises(NotPredictionAlg):
            rs.fit_predict('A000', filter_list=filter_list)

        result_rank_filtered = rs.fit_rank('A000', filter_list=filter_list)
        self.assertEqual(len(result_rank_filtered), 2)

        # Test top-n ranking with the Page Rank algorithm
        result_rank_numbered = rs.fit_rank('A000', recs_number=recs_number)
        self.assertEqual(len(result_rank_numbered), recs_number)

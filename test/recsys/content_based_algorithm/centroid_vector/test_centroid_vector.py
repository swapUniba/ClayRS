import os
from unittest import TestCase

import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import OnlyNegativeItems, NoRatedItems, NotPredictionAlg
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')


class TestCentroidVector(TestCase):

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

        self.filter_list = ['tt0112641', 'tt0112760', 'tt0112896']

        self.movies_dir = os.path.join(contents_path, 'movies_codified/')

    def test_predict(self):
        alg = CentroidVector({'Genre': ['embedding']}, CosineSimilarity(), threshold=0)
        user_ratings = self.ratings.query('from_id == "A000"')
        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict(user_ratings, self.movies_dir)

    def test_rank_single_representation(self):
        # Single representation
        alg = CentroidVector({'Genre': ['embedding']}, CosineSimilarity(), threshold=0)

        user_ratings = self.ratings.query('from_id == "A000"')

        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.movies_dir, filter_list=self.filter_list)
        item_ranked_set = set(res_filtered['to_id'])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.movies_dir)
        item_rated_set = set(user_ratings['to_id'])
        item_ranked_set = set(res_all_unrated['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.movies_dir, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set(user_ratings['to_id'])
        item_ranked_set = set(res_n_recs['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    def test_rank_multiple_representations(self):
        # Multiple representations with auto threshold based on the mean ratings of the user
        alg = CentroidVector({'Plot': ['tfidf', 'embedding'],
                              "Genre": ['tfidf', 'embedding'],
                              'imdbRating': [0]}, CosineSimilarity())

        user_ratings = self.ratings.query('from_id == "A000"')

        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.movies_dir, filter_list=self.filter_list)
        item_ranked_set = set(res_filtered['to_id'])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.movies_dir)
        item_rated_set = set(user_ratings['to_id'])
        item_ranked_set = set(res_all_unrated['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.movies_dir, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set(user_ratings['to_id'])
        item_ranked_set = set(res_n_recs['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    def test_raise_errors(self):
        # Only negative available
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = CentroidVector({'Plot': 'embedding'}, CosineSimilarity(), 0)
        user_ratings = self.ratings.query('from_id == "A000"')

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_ratings, self.movies_dir)

        # No Item avilable locally
        self.ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = CentroidVector({'Plot': 'embedding'}, CosineSimilarity(), 0)
        user_ratings = self.ratings.query('from_id == "A000"')

        with self.assertRaises(NoRatedItems):
            alg.process_rated(user_ratings, self.movies_dir)

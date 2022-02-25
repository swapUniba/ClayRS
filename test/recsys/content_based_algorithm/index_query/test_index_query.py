import os
import unittest
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsIndex
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg, OnlyNegativeItems, \
    NoRatedItems, EmptyUserRatings
from orange_cb_recsys.recsys.content_based_algorithm.index_query.index_query import IndexQuery
from test import dir_test_files


class TestIndexQuery(TestCase):

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
            columns=["user_id", "to_id", "score", "timestamp"])
        self.ratings = Ratings.from_dataframe(ratings)

        self.filter_list = ['tt0112641', 'tt0112760', 'tt0112896']

        index_path = os.path.join(dir_test_files, 'complex_contents', 'index')

        self.available_loaded_items = LoadedContentsIndex(index_path)

    def test_predict(self):
        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict(user_ratings, self.available_loaded_items)

    def test_rank_single_representation(self):
        # Test single representation
        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_filtered])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.available_loaded_items, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_n_recs])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    def test_rank_multiple_representations(self):
        # Multiple representations with auto threshold based on the mean ratings of the user
        alg = IndexQuery({'Plot': ['index_original', 'index_preprocessed'],
                          'Genre': ['index_original', 3]})
        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_filtered])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.available_loaded_items, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_predicted.item_id for interaction_predicted in res_n_recs])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    def test_raise_errors(self):
        # Only negative available
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["user_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_ratings = ratings.get_user_interactions('A000')

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_ratings, self.available_loaded_items)

        # No Item available locally
        ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 1, "54654675")],
            columns=["user_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_ratings = ratings.get_user_interactions('A000')

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_ratings, self.available_loaded_items)

        # User has no ratings
        user_ratings = []

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)

        with self.assertRaises(EmptyUserRatings):
            alg.process_rated(user_ratings, self.available_loaded_items)

        with self.assertRaises(EmptyUserRatings):
            alg.rank(user_ratings, self.available_loaded_items)


if __name__ == '__main__':
    unittest.main()

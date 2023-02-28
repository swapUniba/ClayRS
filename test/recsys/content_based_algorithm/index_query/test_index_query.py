import os
import unittest
from unittest import TestCase
import pandas as pd

from clayrs.content_analyzer import Ratings
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsIndex
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg, OnlyNegativeItems, EmptyUserRatings
from clayrs.recsys.content_based_algorithm.index_query.index_query import IndexQuery
from test import dir_test_files


class TestIndexQuery(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
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
        ratings = Ratings.from_dataframe(ratings)
        ratings.item_map.append(['tt0112641', 'tt0112760'])
        cls.ratings = ratings

        index_path = os.path.join(dir_test_files, 'complex_contents', 'index')

        cls.filter_list = ['tt0112641', 'tt0112760', 'tt0112896']

        cls.available_loaded_items = LoadedContentsIndex(index_path)

    def test_predict(self):
        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict_single_user(user_idx, self.ratings, self.available_loaded_items, self.filter_list)

    def test_rank_single_representation(self):
        # Test single representation
        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # rank with filter_list
        res_filtered = alg.rank_single_user(user_idx, self.ratings, self.available_loaded_items,
                                            recs_number=None, filter_list=self.filter_list)
        # convert int to string for comparison with filter list
        item_ranked_set = set(self.ratings.item_map.convert_seq_int2str(res_filtered[:, 1].astype(int)))
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank with n_recs specified
        n_recs = 2
        res_n_recs = alg.rank_single_user(user_idx, self.ratings, self.available_loaded_items, n_recs, self.filter_list)
        self.assertEqual(len(res_n_recs), n_recs)

    def test_rank_multiple_representations(self):
        # Multiple representations with auto threshold based on the mean ratings of the user
        alg = IndexQuery({'Plot': ['index_original', 'index_preprocessed'],
                          'Genre': ['index_original', 3]})
        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # rank with filter_list
        res_filtered = alg.rank_single_user(user_idx, self.ratings, self.available_loaded_items,
                                            recs_number=None, filter_list=self.filter_list)
        # convert int to string for comparison with filter list
        item_ranked_set = set(self.ratings.item_map.convert_seq_int2str(res_filtered[:, 1].astype(int)))
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank with n_recs specified
        n_recs = 2
        res_n_recs = alg.rank_single_user(user_idx, self.ratings, self.available_loaded_items, n_recs, self.filter_list)
        self.assertEqual(len(res_n_recs), n_recs)

    def test_raise_errors(self):
        # Only negative available
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["user_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # No Item available locally
        ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 1, "54654675")],
            columns=["user_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # User has no ratings
        ratings = Ratings.from_list([('u1', 'i1', 2)], {'u1': 0}, {'i1': 0})
        user_idx = 1

        alg = IndexQuery({'Plot': 'index_original'}, threshold=0)

        with self.assertRaises(EmptyUserRatings):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        with self.assertRaises(EmptyUserRatings):
            alg.rank_single_user(user_idx, ratings, self.available_loaded_items, 2, self.filter_list)


if __name__ == '__main__':
    unittest.main()

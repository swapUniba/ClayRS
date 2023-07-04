import unittest
from unittest import TestCase

from clayrs.content_analyzer import Ratings
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from clayrs.recsys.content_based_algorithm.regressor.linear_predictor import LinearPredictor

from clayrs.recsys.content_based_algorithm.regressor.regressors import SkLinearRegression, \
    SkSGDRegressor, SkARDRegression, SkHuberRegressor, SkPassiveAggressiveRegressor, SkBayesianRidge, SkRidge, \
    Regressor
import os
import pandas as pd

from test import dir_test_files


def for_each_model(test_func):
    def wrapper(self, *args, **kwargs):
        for model in self.models_list:
            with self.subTest(current_model=model):
                test_func(*((self, model) + args), **kwargs)

    return wrapper


class TestRegression(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 3.5, "54654675"),
            ("A000", "tt0112302", 3.5, "54654675"),
            ("A001", "tt0114576", 4, "54654675"),
            ("A001", "tt0112896", 2, "54654675"),
            ("A000", "tt0112346", 2, "54654675"),
            ("A000", "tt0112453", 1, "54654675"),
            ("A002", "tt0112453", 1, "54654675"),
            ("A002", "tt0113497", 3.5, "54654675"),
            ("A003", "tt0112453", 1, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)
        ratings.item_map.append(['tt0112641', 'tt0112760'])
        cls.ratings = ratings

        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
        cls.filter_list = ['tt0112641', 'tt0112760', 'tt0112896', 'tt0113497']

        cls.available_loaded_items = LoadedContentsDict(movies_dir)

        # IMPORTANT! All models to test. If another model is implemented, append it to this list
        cls.models_list = [SkLinearRegression(), SkRidge(), SkBayesianRidge(),
                           SkSGDRegressor(), SkARDRegression(), SkHuberRegressor(),
                           SkPassiveAggressiveRegressor()]

    @for_each_model
    def test_predict_single_representation(self, model: Regressor):
        lm = model

        # Single representation
        alg = LinearPredictor({'Plot': ['tfidf']}, lm)

        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # predict with filter_list
        res_filtered = alg.predict_single_user(user_idx, self.ratings, self.available_loaded_items,
                                               filter_list=self.filter_list)
        # convert int to string for comparison with filter list
        item_scored_set = set(self.ratings.item_map.convert_seq_int2str(res_filtered[:, 1].astype(int)))
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

    @for_each_model
    def test_predict_multiple_representations(self, model: Regressor):
        lm = model

        # Multiple representations filtered only items with score >= 2
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding'],
                               'Genre': ['tfidf', 'embedding'],
                               'imdbRating': [0]}, lm, only_greater_eq=2)

        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # predict with filter_list
        res_filtered = alg.predict_single_user(user_idx, self.ratings, self.available_loaded_items,
                                               filter_list=self.filter_list)
        item_scored_set = set(self.ratings.item_map.convert_seq_int2str(res_filtered[:, 1].astype(int)))
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

    @for_each_model
    def test_rank_single_representation(self, model: Regressor):
        lm = model

        # Single representation
        alg = LinearPredictor({'Plot': ['tfidf']}, lm)

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

    @for_each_model
    def test_rank_multiple_representations(self, model: Regressor):
        lm = model

        # Multiple representations filtered only items with score >= 2
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding'],
                               'Genre': ['tfidf', 'embedding'],
                               'imdbRating': [0]}, lm, only_greater_eq=2)

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
        # No Item available locally
        ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 1, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = LinearPredictor({'Plot': ['tfidf']}, SkLinearRegression())
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(NoRatedItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # User has no ratings
        ratings = Ratings.from_list([('u1', 'i1', 2)], {'u1': 0}, {'i1': 0})
        user_idx = 1

        alg = LinearPredictor({'Plot': ['tfidf']}, SkLinearRegression())

        with self.assertRaises(EmptyUserRatings):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        with self.assertRaises(EmptyUserRatings):
            alg.rank_single_user(user_idx, ratings, self.available_loaded_items, 2, self.filter_list)


if __name__ == '__main__':
    unittest.main()

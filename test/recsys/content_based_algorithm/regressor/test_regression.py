import unittest
from unittest import TestCase

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from orange_cb_recsys.recsys.content_based_algorithm.regressor.linear_predictor import LinearPredictor

from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import SkLinearRegression, \
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
        cls.ratings = Ratings.from_dataframe(ratings)

        cls.filter_list = ['tt0112641', 'tt0112760', 'tt0112896', 'tt0113497']

        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

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

        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # predict with filter_list
        res_filtered = alg.predict(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_scored_set = set([interaction_filtered.item_id for interaction_filtered in res_filtered])
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

        # predict without filter_list
        res_all_unrated = alg.predict(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_scored_set = set([interaction_all.item_id for interaction_all in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_scored = item_scored_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_scored), 0)

    @for_each_model
    def test_predict_multiple_representations(self, model: Regressor):
        lm = model

        # Multiple representations filtered only items with score >= 2
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding'],
                               'Genre': ['tfidf', 'embedding'],
                               'imdbRating': [0]}, lm, only_greater_eq=2)

        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # predict with filter_list
        res_filtered = alg.predict(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_scored_set = set([interaction_filtered.item_id for interaction_filtered in res_filtered])
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

        # predict without filter_list
        res_all_unrated = alg.predict(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_scored_set = set([interaction_all.item_id for interaction_all in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_scored = item_scored_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_scored), 0)

    @for_each_model
    def test_rank_single_representation(self, model: Regressor):
        lm = model

        # Single representation
        alg = LinearPredictor({'Plot': ['tfidf']}, lm)

        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_ranked_set = set([interaction_filtered.item_id for interaction_filtered in res_filtered])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_all.item_id for interaction_all in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.available_loaded_items, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_nrecs.item_id for interaction_nrecs in res_n_recs])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    @for_each_model
    def test_rank_multiple_representations(self, model: Regressor):
        lm = model

        # Multiple representations filtered only items with score >= 2
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding'],
                               'Genre': ['tfidf', 'embedding'],
                               'imdbRating': [0]}, lm, only_greater_eq=2)

        user_ratings = self.ratings.get_user_interactions("A000")

        alg.process_rated(user_ratings, self.available_loaded_items)
        alg.fit()

        # rank with filter_list
        res_filtered = alg.rank(user_ratings, self.available_loaded_items, filter_list=self.filter_list)
        item_ranked_set = set([interaction_filtered.item_id for interaction_filtered in res_filtered])
        self.assertEqual(len(item_ranked_set), len(self.filter_list))
        self.assertCountEqual(item_ranked_set, self.filter_list)

        # rank without filter_list
        res_all_unrated = alg.rank(user_ratings, self.available_loaded_items)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_all.item_id for interaction_all in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

        # rank with n_recs specified
        n_recs = 5
        res_n_recs = alg.rank(user_ratings, self.available_loaded_items, n_recs)
        self.assertEqual(len(res_n_recs), n_recs)
        item_rated_set = set([interaction.item_id for interaction in user_ratings])
        item_ranked_set = set([interaction_nrecs.item_id for interaction_nrecs in res_all_unrated])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_ranked = item_ranked_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_ranked), 0)

    def test_raise_errors(self):
        # No Item available locally
        ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 1, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = LinearPredictor({'Plot': ['tfidf']}, SkLinearRegression())
        user_ratings = ratings.get_user_interactions('A000')

        with self.assertRaises(NoRatedItems):
            alg.process_rated(user_ratings, self.available_loaded_items)

        # User has no ratings
        user_ratings = []

        alg = LinearPredictor({'Plot': ['tfidf']}, SkLinearRegression())

        with self.assertRaises(EmptyUserRatings):
            alg.process_rated(user_ratings, self.available_loaded_items)

        with self.assertRaises(EmptyUserRatings):
            alg.rank(user_ratings, self.available_loaded_items)


if __name__ == '__main__':
    unittest.main()

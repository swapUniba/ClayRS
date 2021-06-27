from unittest import TestCase

from orange_cb_recsys.recsys.content_based_algorithm.regressor.linear_predictor import LinearPredictor

from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import SkLinearRegression, \
    SkSGDRegressor, SkARDRegression, SkHuberRegressor, SkPassiveAggressiveRegressor, SkBayesianRidge, SkRidge, \
    Regressor
from orange_cb_recsys.utils.const import root_path
import os
import pandas as pd

contents_path = os.path.join(root_path, 'contents')


def for_each_model(test_func):
    def wrapper(self, *args, **kwargs):
        for model in self.models_list:
            with self.subTest(current_model=model):
                test_func(*((self, model) + args), **kwargs)

    return wrapper


class TestRegression(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 3.5, "54654675"),
            ("A000", "tt0112302", 3.5, "54654675"),
            ("A001", "tt0114576", 4, "54654675"),
            ("A001", "tt0112896", 2, "54654675"),
            ("A000", "tt0112346", 2, "54654675"),
            ("A000", "tt0112453", 1, "54654675"),
            ("A002", "tt0112453", 1, "54654675"),
            ("A002", "tt0113497", 3.5, "54654675"),
            ("A003", "tt0112453", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        cls.filter_list = ['tt0112641', 'tt0112760', 'tt0112896', 'tt0113497']

        cls.movies_dir = os.path.join(contents_path, 'movies_codified/')

        # IMPORTANT! All models to test. If another model is implemented, append it to this list
        cls.models_list = [SkLinearRegression(), SkRidge(), SkBayesianRidge(),
                           SkSGDRegressor(), SkARDRegression(), SkHuberRegressor(),
                           SkPassiveAggressiveRegressor()]

    @for_each_model
    def test_predict_single_representation(self, model: Regressor):
        lm = model

        # Single representation
        alg = LinearPredictor({'Plot': ['tfidf']}, lm)

        user_ratings = self.ratings.query('from_id == "A000"')

        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # predict with filter_list
        res_filtered = alg.rank(user_ratings, self.movies_dir, filter_list=self.filter_list)
        item_scored_set = set(res_filtered['to_id'])
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

        # predict without filter_list
        res_all_unrated = alg.rank(user_ratings, self.movies_dir)
        item_rated_set = set(user_ratings['to_id'])
        item_scored_set = set(res_all_unrated['to_id'])
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

        user_ratings = self.ratings.query('from_id == "A000"')

        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # predict with filter_list
        res_filtered = alg.rank(user_ratings, self.movies_dir, filter_list=self.filter_list)
        item_scored_set = set(res_filtered['to_id'])
        self.assertEqual(len(item_scored_set), len(self.filter_list))
        self.assertCountEqual(item_scored_set, self.filter_list)

        # predict without filter_list
        res_all_unrated = alg.rank(user_ratings, self.movies_dir)
        item_rated_set = set(user_ratings['to_id'])
        item_scored_set = set(res_all_unrated['to_id'])
        # We expect this to be empty, since the alg should rank only unrated items (unless in filter list)
        rated_in_scored = item_scored_set.intersection(item_rated_set)
        self.assertEqual(len(rated_in_scored), 0)

    @for_each_model
    def test_rank_single_representation(self, model: Regressor):
        lm = model

        # Single representation
        alg = LinearPredictor({'Plot': ['tfidf']}, lm)

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

    @for_each_model
    def test_rank_multiple_representations(self, model: Regressor):
        lm = model

        # Multiple representations filtered only items with score >= 2
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding'],
                               'Genre': ['tfidf', 'embedding'],
                               'imdbRating': [0]}, lm, only_greater_eq=2)

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

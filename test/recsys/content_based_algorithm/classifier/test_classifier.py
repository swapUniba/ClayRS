from unittest import TestCase

import pandas as pd
import os

from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifier_recommender import ClassifierRecommender
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import Classifier,\
    SkGaussianProcess, SkRandomForest, SkLogisticRegression, SkKNN, SkSVC, SkDecisionTree
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import OnlyNegativeItems, NoRatedItems, \
    OnlyPositiveItems, NotPredictionAlg
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')


def for_each_classifier(test_func):
    def wrapper(self, *args, **kwargs):
        for classifier in self.classifiers_list:
            with self.subTest(current_classifier=classifier):
                test_func(*((self, classifier) + args), **kwargs)

    return wrapper


class TestClassifierRecommender(TestCase):

    def setUp(self) -> None:
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 0.5, "54654675"),
            ("A000", "tt0112302", 0.5, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0112346", -0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        # tt0112281 is rated for A000 but let's suppose we want to know its rank
        self.filter_list = ['tt0112281', 'tt0112760', 'tt0112896']

        self.movies_dir = os.path.join(contents_path, 'movies_codified/')

        # IMPORTANT! If a new classifier is added, just add it to this list to test it
        self.classifiers_list = [
            SkSVC(), SkKNN(), SkRandomForest(), SkLogisticRegression(),
            SkDecisionTree(), SkGaussianProcess()
        ]

    def test_predict(self):
        # Doesn't matter which classifier we chose
        alg = ClassifierRecommender({'Plot': ['tfidf']}, SkSVC(), threshold=0)
        user_ratings = self.ratings.query('from_id == "A000"')
        alg.process_rated(user_ratings, self.movies_dir)
        alg.fit()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict(user_ratings, self.movies_dir)

    @for_each_classifier
    def test_rank_single_representation(self, classifier: Classifier):
        clf = classifier

        # Single representation
        alg = ClassifierRecommender({'Plot': ['tfidf']}, clf, threshold=0)

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

    @for_each_classifier
    def test_rank_multiple_representations(self, classifier: Classifier):
        clf = classifier

        # Multiple representations with auto threshold based on the mean ratings of the user
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding'],
                                     'Genre': ['tfidf', 'embedding'],
                                     'imdbRating': [0]}, clf)

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
        # Only positive available
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_ratings = self.ratings.query('from_id == "A000"')

        with self.assertRaises(OnlyPositiveItems):
            alg.process_rated(user_ratings, self.movies_dir)

        # Only negative available
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_ratings = self.ratings.query('from_id == "A000"')

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_ratings, self.movies_dir)

        # No Item avilable locally
        self.ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_ratings = self.ratings.query('from_id == "A000"')

        with self.assertRaises(NoRatedItems):
            alg.process_rated(user_ratings, self.movies_dir)
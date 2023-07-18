import unittest
from unittest import TestCase

import pandas as pd
import os

from clayrs.content_analyzer import Ratings
from clayrs.recsys.content_based_algorithm.classifier.classifier_recommender import ClassifierRecommender
from clayrs.recsys.content_based_algorithm.classifier.classifiers import Classifier,\
    SkGaussianProcess, SkRandomForest, SkLogisticRegression, SkKNN, SkSVC, SkDecisionTree
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.content_based_algorithm.exceptions import OnlyNegativeItems, NoRatedItems, \
    OnlyPositiveItems, NotPredictionAlg, EmptyUserRatings
from test import dir_test_files


def for_each_classifier(test_func):
    def wrapper(self, *args, **kwargs):
        for classifier in self.classifiers_list:
            with self.subTest(current_classifier=classifier):
                test_func(*((self, classifier) + args), **kwargs)

    return wrapper


class TestClassifierRecommender(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 0.5, "54654675"),
            ("A000", "tt0112302", 0.5, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0112346", -0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)
        ratings.item_map.append(['tt0112760'])
        cls.ratings = ratings

        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
        # tt0112281 is rated for A000 but let's suppose we want to know its rank
        cls.filter_list = ['tt0112281', 'tt0112760', 'tt0112896']

        cls.available_loaded_items = LoadedContentsDict(movies_dir)

        # IMPORTANT! If a new classifier is added, just add it to this list to test it
        cls.classifiers_list = [
            SkSVC(), SkKNN(), SkRandomForest(), SkLogisticRegression(),
            SkDecisionTree(), SkGaussianProcess()
        ]

    def test_predict(self):
        # Doesn't matter which classifier we chose
        alg = ClassifierRecommender({'Plot': ['tfidf']}, SkSVC(), threshold=0)
        user_idx = self.ratings.user_map['A000']
        alg.process_rated(user_idx, self.ratings, self.available_loaded_items)
        alg.fit_single_user()

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict_single_user(user_idx, self.ratings, self.available_loaded_items, self.filter_list)

    @for_each_classifier
    def test_rank_single_representation(self, classifier: Classifier):
        clf = classifier

        # Single representation
        alg = ClassifierRecommender({'Plot': ['tfidf']}, clf, threshold=0)

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

    @for_each_classifier
    def test_rank_multiple_representations(self, classifier: Classifier):
        clf = classifier

        # Multiple representations with auto threshold based on the mean ratings of the user
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding'],
                                     'Genre': ['tfidf', 'embedding'],
                                     'imdbRating': [0]}, clf)

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
        # Only positive available
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 1, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(OnlyPositiveItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # Only negative available
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(OnlyNegativeItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # No Item avilable locally
        ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 0.5, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(ratings)

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)
        user_idx = self.ratings.user_map['A000']

        with self.assertRaises(NoRatedItems):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        # User has no ratings
        ratings = Ratings.from_list([('u1', 'i1', 2)], {'u1': 0}, {'i1': 0})
        user_idx = 1

        alg = ClassifierRecommender({'Plot': 'tfidf'}, SkKNN(), 0)

        with self.assertRaises(EmptyUserRatings):
            alg.process_rated(user_idx, ratings, self.available_loaded_items)

        with self.assertRaises(EmptyUserRatings):
            alg.rank_single_user(user_idx, ratings, self.available_loaded_items, 2, self.filter_list)


if __name__ == '__main__':
    unittest.main()

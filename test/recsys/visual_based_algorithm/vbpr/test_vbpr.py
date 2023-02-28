import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from clayrs.content_analyzer import Ratings
from clayrs.recsys import TestRatingsMethodology
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.visual_based_algorithm.vbpr.vbpr_algorithm import VBPR
from test import dir_test_files

train_ratings = pd.DataFrame.from_records([
    ("A000", "tt0114576", 5, "54654675"),
    ("A001", "tt0114576", 3, "54654675"),
    ("A001", "tt0112896", 1, "54654675"),
    ("A000", "tt0113041", 1, "54654675"),
    ("A002", "tt0112453", 2, "54654675"),
    ("A000", "non_existent", 2, "54654675"),
    ("A002", "tt0113497", 4, "54654675"),
    ("A003", "tt0112453", 1, "54654675"),
    ("A003", "tt0113497", 4, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

test_ratings = pd.DataFrame.from_records([
    ("A000", "tt0114388", None),
    ("A000", "tt0112302", None),
    ("A001", "tt0113189", None),
    ("A001", "tt0113228", None),
    ("A002", "tt0114319", None),
    ("A002", "tt0114709", None),
    ("A003", "tt0114885", None)],
    columns=["from_id", "to_id", "score"])

# we create manually the mapping since we want a global mapping containing train and test items
user_map = {}
users_train = train_ratings["from_id"]
users_test = test_ratings["from_id"]

all_users = users_train.append(users_test)
for user_id in all_users:
    if user_id not in user_map:
        user_map[user_id] = len(user_map)

item_map = {}

items_train = train_ratings["to_id"]
items_test = test_ratings["to_id"]

all_items = items_train.append(items_test)
for item_id in all_items:
    if item_id not in item_map:
        item_map[item_id] = len(item_map)


class TestVBPR(TestCase):
    train_ratings = Ratings.from_dataframe(train_ratings, user_map=user_map, item_map=item_map)
    test_ratings = Ratings.from_dataframe(test_ratings, user_map=user_map, item_map=item_map)

    user_map = train_ratings.user_map
    item_map = train_ratings.item_map

    movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

    def test_predict(self):
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu", seed=42)
        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.predict(alg_network, train_set=self.train_ratings, test_set=self.train_ratings,
                        items_directory=self.movies_dir, user_idx_list=self.test_ratings.unique_user_idx_column,
                        methodology=TestRatingsMethodology(), num_cpus=1)

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.fit_predict(train_set=self.train_ratings, test_set=self.test_ratings, items_directory=self.movies_dir,
                            user_idx_list=self.test_ratings.unique_user_idx_column,
                            methodology=TestRatingsMethodology(), num_cpus=1, save_fit=True)

    def test_build_only_positive_ratings(self):

        # ------ FIXED THRESHOLD (but still all ratings are returned) ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=0, seed=42)

        only_pos_ratings = alg._build_only_positive_ratings(self.train_ratings)

        self.assertEqual(self.train_ratings, only_pos_ratings)

        # ------ FIXED THRESHOLD (for each user, a subset of its ratings is returned) ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=3, seed=42)

        expected_pos_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 5, "54654675"),
            ("A001", "tt0114576", 3, "54654675"),
            ("A002", "tt0113497", 4, "54654675"),
            ("A003", "tt0113497", 4, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        expected_pos_ratings = Ratings.from_dataframe(expected_pos_ratings,
                                                      user_map=self.train_ratings.user_map,
                                                      item_map=self.train_ratings.item_map)

        only_pos_ratings = alg._build_only_positive_ratings(self.train_ratings)

        self.assertEqual(expected_pos_ratings, only_pos_ratings)

        # ------ FIXED THRESHOLD (for some users, no ratings are returned) ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=4, seed=42)

        # user 1 doesn't have any rating anymore
        expected_pos_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 5, "54654675"),
            ("A002", "tt0113497", 4, "54654675"),
            ("A003", "tt0113497", 4, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        expected_pos_ratings = Ratings.from_dataframe(expected_pos_ratings,
                                                      user_map=self.train_ratings.user_map,
                                                      item_map=self.train_ratings.item_map)

        only_pos_ratings = alg._build_only_positive_ratings(self.train_ratings)

        self.assertEqual(expected_pos_ratings, only_pos_ratings)

        # --- FIXED THRESHOLD (empty ratings are returned, meaning that the threshold was too high for all users) ---
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=6, seed=42)

        with self.assertRaises(ValueError):
            alg._build_only_positive_ratings(self.train_ratings)

        # ------ THRESHOLD SET TO NONE (for each user, a subset of its ratings is returned) ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=None, seed=42)

        expected_pos_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 5, "54654675"),
            ("A001", "tt0114576", 3, "54654675"),
            ("A002", "tt0113497", 4, "54654675"),
            ("A003", "tt0113497", 4, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        expected_pos_ratings = Ratings.from_dataframe(expected_pos_ratings,
                                                      user_map=self.train_ratings.user_map,
                                                      item_map=self.train_ratings.item_map)

        only_pos_ratings = alg._build_only_positive_ratings(self.train_ratings)

        self.assertEqual(expected_pos_ratings, only_pos_ratings)

    def test_load_items_features(self):

        # ------ Generic case, items features (for items in ratings) are loaded from the source ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=None, normalize=False, seed=42)

        features = alg._load_items_features(self.train_ratings, self.movies_dir)

        items_id_locally_available = set(filename.split(".")[0] for filename in os.listdir(self.movies_dir))
        items_id_to_extract_features = list(self.train_ratings.item_map.map)

        first_not_none_element = next(item for item in features if item is not None)
        for i, item_id_to_extract in enumerate(items_id_to_extract_features):

            # if is not present then array of zeros is used
            if item_id_to_extract not in items_id_locally_available:
                expected = torch.zeros(size=first_not_none_element.shape)
                result = features[i]

                np.testing.assert_array_equal(expected.numpy(), result)
            # otherwise if it is present obviously we expect that features are different from array of zeros
            else:
                not_expected = torch.zeros(size=first_not_none_element.shape)
                result = features[i]

                self.assertFalse(np.array_equal(not_expected.numpy(), result))

        # ------ Normalization set to True, features are expected to be in the [0, 1] range ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=None, normalize=True, seed=42)

        features = alg._load_items_features(self.train_ratings, self.movies_dir)

        self.assertTrue(all(min(feature) >= 0 and max(feature) <= 1 for feature in features))

        # ------ Limit case, all items in the ratings do not have a matching serialized feature ------
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=0, device="cpu",
                   threshold=None, seed=42)

        ratings_non_existent_features_items = pd.DataFrame.from_records([
            ("A000", "featureless_1", 4, "54654675"),
            ("A001", "featureless_2", 3, "54654675"),
            ("A002", "featureless_3", 5, "54654675"),
            ("A002", "featureless_4", 4, "54654675"),
            ("A003", "featureless_5", 4, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings_non_existent_features_items = Ratings.from_dataframe(ratings_non_existent_features_items)

        with self.assertRaises(FileNotFoundError):
            alg._load_items_features(ratings_non_existent_features_items, self.movies_dir)

    def test_rank_single_representation(self):
        # Single representation
        alg = VBPR({'Genre': ['embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=1, device="cpu", seed=42)
        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        # unbounded rank
        result_rank_filtered = alg.rank(alg_network, train_set=self.train_ratings, test_set=self.test_ratings,
                                        items_directory=self.movies_dir,
                                        user_idx_list=self.train_ratings.unique_user_idx_column,
                                        methodology=TestRatingsMethodology(), num_cpus=1, n_recs=None)

        # result_rank_filtered contains one ranked uir per user
        self.assertEqual(len(self.test_ratings.unique_user_idx_column), len(result_rank_filtered))

        for rank_user_uir in result_rank_filtered:
            user_idx = rank_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(self.test_ratings.get_user_interactions(user_idx)), len(rank_user_uir))

        # rank with recs_number specified
        n_recs = 2
        res_n_recs = alg.rank(alg_network, train_set=self.train_ratings, test_set=self.test_ratings,
                              items_directory=self.movies_dir, user_idx_list=self.test_ratings.unique_user_idx_column,
                              methodology=TestRatingsMethodology(), num_cpus=1, n_recs=n_recs)

        # result_rank_filtered contains one ranked uir per user
        self.assertEqual(len(self.test_ratings.unique_user_idx_column), len(result_rank_filtered))

        for rank_user_uir in res_n_recs[:-1]:
            self.assertEqual(n_recs, len(rank_user_uir))

        # user A003 has only 1 item in the test set to recommend (test ratings methodology)
        self.assertEqual(1, len(res_n_recs[-1]))

    def test_rank_multiple_representations(self):
        # Multiple representations
        alg = VBPR({'Plot': ['tfidf', 'embedding'], 'Genre': ['tfidf', 'embedding']}, gamma_dim=10, theta_dim=10,
                   batch_size=64, epochs=1, device="cpu", seed=42)
        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        # unbounded rank
        result_rank_filtered = alg.rank(alg_network, train_set=self.train_ratings, test_set=self.test_ratings,
                                        items_directory=self.movies_dir,
                                        user_idx_list=self.test_ratings.unique_user_idx_column,
                                        methodology=TestRatingsMethodology(), num_cpus=1, n_recs=None)

        # result_rank_filtered contains one ranked uir per user
        self.assertEqual(len(self.test_ratings.unique_user_idx_column), len(result_rank_filtered))

        for rank_user_uir in result_rank_filtered:
            user_idx = rank_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(self.test_ratings.get_user_interactions(user_idx)), len(rank_user_uir))

        # rank with recs_number specified
        n_recs = 2
        res_n_recs = alg.rank(alg_network, train_set=self.train_ratings, test_set=self.test_ratings,
                              items_directory=self.movies_dir, user_idx_list=self.test_ratings.unique_user_idx_column,
                              methodology=TestRatingsMethodology(), num_cpus=1, n_recs=n_recs)

        # result_rank_filtered contains one ranked uir per user
        self.assertEqual(len(self.test_ratings.unique_user_idx_column), len(result_rank_filtered))

        for rank_user_uir in res_n_recs[:-1]:
            self.assertEqual(n_recs, len(rank_user_uir))

        # user A003 has only 1 item in the test set to recommend (test ratings methodology)
        self.assertEqual(1, len(res_n_recs[-1]))

    def test_fit_rank(self):
        # Compare results when using fit and rank with results obtained from calling fit_rank directly
        alg = VBPR({'Genre': ['tfidf', 'embedding']}, gamma_dim=10, theta_dim=10, batch_size=64, epochs=1,
                   device="cpu", seed=42)

        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)
        result_rank_fit = alg.rank(alg_network, train_set=self.train_ratings, test_set=self.test_ratings,
                                   items_directory=self.movies_dir,
                                   user_idx_list=self.test_ratings.unique_user_idx_column,
                                   methodology=TestRatingsMethodology(), num_cpus=1, n_recs=None)

        alg_network_fit, result_rank_fit_rank = alg.fit_rank(train_set=self.train_ratings, test_set=self.test_ratings,
                                                             items_directory=self.movies_dir,
                                                             user_idx_list=self.test_ratings.unique_user_idx_column,
                                                             methodology=TestRatingsMethodology(), num_cpus=1,
                                                             n_recs=None,
                                                             save_fit=True)

        for single_result_rank_fit, single_result_rank_fit_rank in zip(result_rank_fit, result_rank_fit_rank):
            np.testing.assert_array_equal(single_result_rank_fit, single_result_rank_fit_rank)

        self.assertIsNotNone(alg_network_fit)  # save_fit == True so the fit alg is returned

        # test save_fit == False
        alg_network_fit, _ = alg.fit_rank(train_set=self.train_ratings, test_set=self.test_ratings,
                                          items_directory=self.movies_dir,
                                          user_idx_list=self.test_ratings.unique_user_idx_column,
                                          methodology=TestRatingsMethodology(), num_cpus=1,
                                          n_recs=None,
                                          save_fit=False)

        self.assertIsNone(alg_network_fit)  # save_fit == False is we don't save the fit alg


if __name__ == '__main__':
    unittest.main()

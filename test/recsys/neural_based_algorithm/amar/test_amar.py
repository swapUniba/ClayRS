import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from clayrs.content_analyzer import Ratings
from clayrs.recsys import TestRatingsMethodology, AmarNetworkBasic, AmarNetworkMerge, AmarNetworkEntityBasedConcat
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.network_based_algorithm.amar.amar_alg import AmarSingleSource, AmarDoubleSource
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

# manually create the mapping to have a global mapping containing train and test items
user_map = {}
users_train = train_ratings["from_id"]
users_test = test_ratings["from_id"]

all_users = pd.concat([users_train, users_test])
for user_id in all_users:
    if user_id not in user_map:
        user_map[user_id] = len(user_map)

item_map = {}

items_train = train_ratings["to_id"]
items_test = test_ratings["to_id"]

all_items = pd.concat([items_train, items_test])
for item_id in all_items:
    if item_id not in item_map:
        item_map[item_id] = len(item_map)


class TestAMAR(TestCase):
    train_ratings: Ratings = Ratings.from_dataframe(train_ratings, user_map=user_map, item_map=item_map)
    test_ratings: Ratings = Ratings.from_dataframe(test_ratings, user_map=user_map, item_map=item_map)

    user_map = train_ratings.user_map
    item_map = train_ratings.item_map

    movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
    users_dir = os.path.join(dir_test_files, 'complex_contents', 'users_codified/')

    def test_predict(self):
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)
        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir)

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

    def test_make_ratings_implicit(self):

        # ------ FIXED THRESHOLD (all positive ratings) ------
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=0)

        expected_implicit_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A001", "tt0114576", 1, "54654675"),
            ("A001", "tt0112896", 1, "54654675"),
            ("A000", "tt0113041", 1, "54654675"),
            ("A002", "tt0112453", 1, "54654675"),
            ("A000", "non_existent", 1, "54654675"),
            ("A002", "tt0113497", 1, "54654675"),
            ("A003", "tt0112453", 1, "54654675"),
            ("A003", "tt0113497", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        expected_implicit_ratings = Ratings.from_dataframe(expected_implicit_ratings,
                                                           user_map=self.train_ratings.user_map,
                                                           item_map=self.train_ratings.item_map)

        implicit_ratings = alg._make_ratings_implicit(self.train_ratings)

        self.assertEqual(expected_implicit_ratings, implicit_ratings)

        # ------ FIXED THRESHOLD ------
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=3)

        expected_implicit_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A001", "tt0114576", 1, "54654675"),
            ("A001", "tt0112896", 0, "54654675"),
            ("A000", "tt0113041", 0, "54654675"),
            ("A002", "tt0112453", 0, "54654675"),
            ("A000", "non_existent", 0, "54654675"),
            ("A002", "tt0113497", 1, "54654675"),
            ("A003", "tt0112453", 0, "54654675"),
            ("A003", "tt0113497", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        expected_implicit_ratings = Ratings.from_dataframe(expected_implicit_ratings,
                                                           user_map=self.train_ratings.user_map,
                                                           item_map=self.train_ratings.item_map)

        implicit_ratings = alg._make_ratings_implicit(self.train_ratings)

        self.assertEqual(expected_implicit_ratings, implicit_ratings)

        # ------ FIXED THRESHOLD (for some users, only negative ratings) ------
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=4)

        # user 1 only has negative ratings
        expected_implicit_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A001", "tt0114576", 0, "54654675"),
            ("A001", "tt0112896", 0, "54654675"),
            ("A000", "tt0113041", 0, "54654675"),
            ("A002", "tt0112453", 0, "54654675"),
            ("A000", "non_existent", 0, "54654675"),
            ("A002", "tt0113497", 1, "54654675"),
            ("A003", "tt0112453", 0, "54654675"),
            ("A003", "tt0113497", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        expected_implicit_ratings = Ratings.from_dataframe(expected_implicit_ratings,
                                                           user_map=self.train_ratings.user_map,
                                                           item_map=self.train_ratings.item_map)

        implicit_ratings = alg._make_ratings_implicit(self.train_ratings)

        self.assertEqual(expected_implicit_ratings, implicit_ratings)

        # --- FIXED THRESHOLD (only negative ratings are returned, meaning that the threshold was too high for all users) ---
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=6)

        expected_implicit_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 0, "54654675"),
            ("A001", "tt0114576", 0, "54654675"),
            ("A001", "tt0112896", 0, "54654675"),
            ("A000", "tt0113041", 0, "54654675"),
            ("A002", "tt0112453", 0, "54654675"),
            ("A000", "non_existent", 0, "54654675"),
            ("A002", "tt0113497", 0, "54654675"),
            ("A003", "tt0112453", 0, "54654675"),
            ("A003", "tt0113497", 0, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        expected_implicit_ratings = Ratings.from_dataframe(expected_implicit_ratings,
                                                           user_map=self.train_ratings.user_map,
                                                           item_map=self.train_ratings.item_map)

        implicit_ratings = alg._make_ratings_implicit(self.train_ratings)

        self.assertEqual(expected_implicit_ratings, implicit_ratings)

        # ------ THRESHOLD SET TO NONE ------
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        expected_implicit_ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A001", "tt0114576", 1, "54654675"),
            ("A001", "tt0112896", 0, "54654675"),
            ("A000", "tt0113041", 0, "54654675"),
            ("A002", "tt0112453", 0, "54654675"),
            ("A000", "non_existent", 0, "54654675"),
            ("A002", "tt0113497", 1, "54654675"),
            ("A003", "tt0112453", 0, "54654675"),
            ("A003", "tt0113497", 1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        expected_implicit_ratings = Ratings.from_dataframe(expected_implicit_ratings,
                                                           user_map=self.train_ratings.user_map,
                                                           item_map=self.train_ratings.item_map)

        implicit_ratings = alg._make_ratings_implicit(self.train_ratings)

        self.assertEqual(expected_implicit_ratings, implicit_ratings)

    def test_load_contents_features(self):

        # Generic case, items features (for items in ratings) are loaded from the source
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        features = alg._load_contents_features(self.train_ratings.item_map, self.movies_dir, {'Genre': ['embedding']})

        items_id_locally_available = set(filename.split(".")[0] for filename in os.listdir(self.movies_dir))
        items_id_to_extract_features = list(self.train_ratings.item_map.map)

        first_not_none_element = next(item for item in features if item is not None)
        for i, item_id_to_extract in enumerate(items_id_to_extract_features):

            # if is not present then array of zeros is used
            if item_id_to_extract not in items_id_locally_available:
                expected = torch.zeros(size=first_not_none_element.shape)
                result = features[i]

                np.testing.assert_array_equal(expected.numpy(), result)
            # otherwise if it is present obviously it is expected that features are different from array of zeros
            else:
                not_expected = torch.zeros(size=first_not_none_element.shape)
                result = features[i]

                self.assertFalse(np.array_equal(not_expected.numpy(), result))

        # Limit case, all items in the ratings do not have a matching serialized feature
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        ratings_non_existent_features_items = pd.DataFrame.from_records([
            ("A000", "featureless_1", 4, "54654675"),
            ("A001", "featureless_2", 3, "54654675"),
            ("A002", "featureless_3", 5, "54654675"),
            ("A002", "featureless_4", 4, "54654675"),
            ("A003", "featureless_5", 4, "54654675")],
            columns=["user_id", "item_id", "score", "timestamp"])
        ratings_non_existent_features_items = Ratings.from_dataframe(ratings_non_existent_features_items)

        with self.assertRaises(FileNotFoundError):
            alg._load_contents_features(ratings_non_existent_features_items.item_map, self.movies_dir, {'Genre': ['embedding']})

    def test_combine_items_features_for_users(self):

        # When user field is not specified, user features should be the centroid of the items liked by the user
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=3)

        implicit_train_set = alg._make_ratings_implicit(self.train_ratings)
        features = alg._load_contents_features(implicit_train_set.item_map, self.movies_dir, {'Genre': ['embedding']})

        users_features = alg._combine_items_features_for_users(implicit_train_set, features)

        item_0_for_user_1 = features[implicit_train_set.item_map.convert_str2int("tt0114576")]
        user_1_features = users_features[implicit_train_set.user_map.convert_str2int("A001")]

        self.assertTrue(np.array_equal(item_0_for_user_1, user_1_features))

        # When user field is not specified, user features should be the centroid of the items liked by the user
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=6)

        implicit_train_set = alg._make_ratings_implicit(self.train_ratings)
        features = alg._load_contents_features(implicit_train_set.item_map, self.movies_dir, {'Genre': ['embedding']})

        users_features = alg._combine_items_features_for_users(implicit_train_set, features)
        user_1_features = users_features[implicit_train_set.user_map.convert_str2int("A001")]

        self.assertFalse(np.any(user_1_features))

        # When user field is not specified, user features should be the centroid of the items liked by the user
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=0)

        implicit_train_set = alg._make_ratings_implicit(self.train_ratings)
        features = alg._load_contents_features(implicit_train_set.item_map, self.movies_dir, {'Genre': ['embedding']})

        users_features = alg._combine_items_features_for_users(implicit_train_set, features)

        item_0_for_user_1 = features[implicit_train_set.item_map.convert_str2int("tt0114576")]
        item_1_for_user_1 = features[implicit_train_set.item_map.convert_str2int("tt0112896")]

        user_1_features = users_features[implicit_train_set.user_map.convert_str2int("A001")]

        centroid = np.vstack([item_0_for_user_1, item_1_for_user_1]).mean(axis=0)

        self.assertTrue(np.array_equal(user_1_features, centroid))

        # When user field is not specified, user features should be the centroid of the items liked by the user
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        implicit_train_set = alg._make_ratings_implicit(self.train_ratings)
        features = alg._load_contents_features(implicit_train_set.item_map, self.movies_dir, {'Genre': ['embedding']})

        users_features = alg._combine_items_features_for_users(implicit_train_set, features)

        item_0_for_user_1 = features[implicit_train_set.item_map.convert_str2int("tt0114576")]

        user_1_features = users_features[implicit_train_set.user_map.convert_str2int("A001")]

        self.assertTrue(np.array_equal(user_1_features, item_0_for_user_1))

        # When user field is not specified, user features should be the centroid of the items liked by the user
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               user_field={'': ''},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        implicit_train_set = alg._make_ratings_implicit(self.train_ratings)
        features = alg._load_contents_features(implicit_train_set.item_map, self.movies_dir, {'Genre': ['embedding']})

        users_features = alg._combine_items_features_for_users(implicit_train_set, features)

        item_0_for_user_1 = features[implicit_train_set.item_map.convert_str2int("tt0114576")]

        user_1_features = users_features[implicit_train_set.user_map.convert_str2int("A001")]

        self.assertTrue(np.array_equal(user_1_features, item_0_for_user_1))

    def test_rank_single_representation(self):
        # Single representation
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)
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
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Plot': ['tfidf', 'embedding'], 'Genre': ['tfidf', 'embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)
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
        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['tfidf', 'embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)

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

        self.assertIsNone(alg_network_fit)  # save_fit == False don't save the fit alg

    def test_fit_users_features(self):

        train_ratings_different_uid = pd.DataFrame.from_records([
            ("0", "tt0114576", 5, "54654675"),
            ("1", "tt0114576", 3, "54654675"),
            ("1", "tt0112896", 1, "54654675"),
            ("0", "tt0113041", 1, "54654675"),
            ("2", "tt0112453", 2, "54654675"),
            ("2", "tt0113497", 4, "54654675"),
            ("3", "tt0112453", 1, "54654675"),
            ("3", "tt0113497", 4, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])
        ratings = Ratings.from_dataframe(train_ratings_different_uid)

        # since no user directory or user field are defined, the algorithm will compute the user representation as the
        # centroid of the items features for which users have given a positive rating

        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['tfidf', 'embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=4)

        alg_network = alg.fit(ratings, items_directory=self.movies_dir, num_cpus=1)

        items_features = alg_network.items_features

        user_representations = []

        for user in ratings.unique_user_idx_column:
            pos_items_interactions = ratings.get_user_interactions(user)
            pos_items_interactions = pos_items_interactions[pos_items_interactions[:, 2] >= 4]
            repr = torch.mean(items_features[pos_items_interactions[:, 1]], dim=0)

            if any(np.isnan(repr)):
                repr = np.full(len(repr), 0)

            user_representations.append(repr)

        user_representations = np.vstack(user_representations)

        np.testing.assert_array_almost_equal(alg_network.users_features.numpy(), user_representations)

        # since a user directory and a user field are defined,
        # the algorithm will load the user representations from memory

        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['tfidf', 'embedding']},
                               user_field={'0': 0},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=4)

        alg_network = alg.fit(ratings, items_directory=self.movies_dir, users_directory=self.users_dir, num_cpus=1)

        features = []

        for i in range(4):
            features.append(np.full((1, 100), i))

        np.testing.assert_array_equal(alg_network.users_features, np.vstack(features))

    def test_load_custom_weights(self):

        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)

        alg_network = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        parameters = []
        for name, parameter in alg_network.named_parameters():
            if parameter.requires_grad and 'weight' in name:
                parameters.append(parameter.detach().numpy())

        # check that parameters passed as argument are the same as the ones initialized

        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None,
                               custom_network_weights=parameters)

        alg_network_custom_w = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        parameters_custom_w = []
        for name, parameter in alg_network_custom_w.named_parameters():
            if parameter.requires_grad and 'weight' in name:
                parameters_custom_w.append(parameter.detach().numpy())

        parameters_custom_w = np.array(parameters_custom_w, dtype=object)

        for param, param_custom in zip(parameters, parameters_custom_w):
            np.testing.assert_array_equal(param, param_custom)

        # check that init parameters are different from custom ones

        alg = AmarSingleSource(network_to_use=AmarNetworkBasic,
                               item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=0,
                               device="cpu",
                               seed=42,
                               threshold=None)

        alg_network_init = alg.fit(self.train_ratings, items_directory=self.movies_dir, num_cpus=1)

        parameters_init = []
        for name, parameter in alg_network_init.named_parameters():
            if parameter.requires_grad and 'weight' in name:
                parameters_init.append(parameter.detach().numpy())

        parameters_init = np.array(parameters_init, dtype=object)

        for param_custom, param_init in zip(parameters_custom_w, parameters_init):
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(param_custom, param_init)

    def test_multiple_sources(self):

        # Multiple AMAR sources use the same code as Single AMAR sources so it is just checked that an algorithm
        # is returned when calling fit rank

        alg = AmarDoubleSource(network_to_use=AmarNetworkEntityBasedConcat,
                               first_item_field={'Genre': ['tfidf', 'embedding']},
                               second_item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)

        alg_network_fit, result_rank_fit_rank = alg.fit_rank(train_set=self.train_ratings, test_set=self.test_ratings,
                                                             items_directory=self.movies_dir,
                                                             user_idx_list=self.test_ratings.unique_user_idx_column,
                                                             methodology=TestRatingsMethodology(), num_cpus=1,
                                                             n_recs=None,
                                                             save_fit=True)

        self.assertIsNotNone(alg_network_fit)  # save_fit == True so the fit alg is returned

        # Multiple AMAR sources use the same code as Single AMAR sources so it is just checked that an algorithm
        # is returned when calling fit rank

        alg = AmarDoubleSource(network_to_use=AmarNetworkMerge,
                               first_item_field={'Genre': ['tfidf', 'embedding']},
                               second_item_field={'Genre': ['embedding']},
                               batch_size=64,
                               epochs=1,
                               device="cpu",
                               seed=42,
                               threshold=None)

        alg_network_fit, result_rank_fit_rank = alg.fit_rank(train_set=self.train_ratings, test_set=self.test_ratings,
                                                             items_directory=self.movies_dir,
                                                             user_idx_list=self.test_ratings.unique_user_idx_column,
                                                             methodology=TestRatingsMethodology(), num_cpus=1,
                                                             n_recs=None,
                                                             save_fit=True)

        self.assertIsNotNone(alg_network_fit)  # save_fit == True so the fit alg is returned


if __name__ == '__main__':
    unittest.main()

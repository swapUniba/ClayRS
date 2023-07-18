import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from clayrs.content_analyzer import Centroid, Ratings
from clayrs.recsys import IndexQuery, LinearPredictor, SkLinearRegression
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict, LoadedContentsIndex
from clayrs.utils.load_content import load_content_instance
from clayrs.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from clayrs.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
from clayrs.recsys.methodology import TestRatingsMethodology, AllItemsMethodology

from test import dir_test_files


train_ratings = pd.DataFrame.from_records([
    ("A000", "tt0114576", 5, "54654675"),
    ("A001", "tt0114576", 3, "54654675"),
    ("A001", "tt0112896", 1, "54654675"),
    ("A000", "tt0113041", 1, "54654675"),
    ("A002", "tt0112453", 2, "54654675"),
    ("A002", "tt0113497", 4, "54654675"),
    ("A003", "tt0112453", 1, "54654675"),
    ("A003", "tt0113497", 4, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

# No locally available items for A000
train_ratings_some_missing = pd.DataFrame.from_records([
    ("A000", "not_existent1", 5, "54654675"),
    ("A001", "tt0114576", 3, "54654675"),
    ("A001", "tt0112896", 1, "54654675"),
    ("A000", "not_existent2", 5, "54654675")],
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
item_map = {}
all_items = train_ratings[["to_id"]].append(test_ratings[["to_id"]]).append(train_ratings_some_missing[["to_id"]])[
    "to_id"]
for item_id in all_items:
    if item_id not in item_map:
        item_map[item_id] = len(item_map)

user_map = {}
all_users = \
    train_ratings[["from_id"]].append(test_ratings[["from_id"]]).append(train_ratings_some_missing[["from_id"]])[
        "from_id"]
for user_id in all_users:
    if user_id not in user_map:
        user_map[user_id] = len(user_map)

train_ratings = Ratings.from_dataframe(train_ratings, user_map=user_map, item_map=item_map)
train_ratings_some_missing = Ratings.from_dataframe(train_ratings_some_missing, user_map=user_map, item_map=item_map)
test_ratings = Ratings.from_dataframe(test_ratings, user_map=user_map, item_map=item_map)


class TestContentBasedAlgorithm(TestCase):

    def setUp(self) -> None:
        # ContentBasedAlgorithm is an abstract class, so we need to instantiate
        # a subclass to test its methods.
        self.alg = CentroidVector({'Plot': 'tfidf'}, CosineSimilarity(), 0)

    def test__bracket_representation(self):
        item_field = {'Plot': 'tfidf',
                      'Genre': [0],
                      'Title': [0, 'trybracket'],
                      'Director': 5}

        item_field_bracketed = {'Plot': ['tfidf'],
                                'Genre': [0],
                                'Title': [0, 'trybracket'],
                                'Director': [5]}

        result = self.alg._bracket_representation(item_field)

        self.assertEqual(item_field_bracketed, result)

    def test_extract_features_item(self):
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

        content = load_content_instance(movies_dir, 'tt0112281')

        result = self.alg.extract_features_item(content)

        self.assertEqual(1, len(result))
        self.assertIsInstance(result[0], sparse.csc_matrix)

    def test_fuse_representations(self):
        dv = DictVectorizer(sparse=False, sort=False)

        tfidf_result1 = {'word1': 1.546, 'word2': 1.467, 'word3': 0.55}
        doc_embedding_result1 = np.array([[0.98347, 1.384038, 7.1023803, 1.09854]])
        word_embedding_result1 = np.array([[0.123, 0.44561], [1.889, 3.22], [0.283, 0.887]])
        float_result1 = 8.8

        tfidf_result2 = {'word2': 1.467, 'word4': 1.1}
        doc_embedding_result2 = np.array([[2.331, 0.887, 1.1123, 0.7765]])
        word_embedding_result2 = np.array([[0.123, 0.44561], [5.554, 1.1234]])
        int_result2 = 7

        x = [[tfidf_result1, doc_embedding_result1, word_embedding_result1, float_result1],
             [tfidf_result2, doc_embedding_result2, word_embedding_result2, int_result2]]

        result = self.alg.fuse_representations(x, Centroid())

        dv.fit([tfidf_result1, tfidf_result2])
        centroid_word_embedding_1 = Centroid().combine(word_embedding_result1)
        centroid_word_embedding_2 = Centroid().combine(word_embedding_result2)

        expected_1 = np.hstack([dv.transform(tfidf_result1).flatten(), doc_embedding_result1.flatten(),
                                centroid_word_embedding_1.flatten(), float_result1])

        expected_2 = np.hstack([dv.transform(tfidf_result2).flatten(), doc_embedding_result2.flatten(),
                                centroid_word_embedding_2.flatten(), int_result2])

        self.assertTrue(all(isinstance(rep, np.ndarray) for rep in result))
        self.assertTrue(np.allclose(result[0], expected_1))
        self.assertTrue(np.allclose(result[1], expected_2))

    def test__load_available_contents(self):
        # test load_available_contents for content based algorithm
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

        interface_dict = self.alg._load_available_contents(movies_dir)
        self.assertIsInstance(interface_dict, LoadedContentsDict)

        interface_dict = self.alg._load_available_contents(movies_dir, {'tt0112281', 'tt0112302'})
        self.assertTrue(len(interface_dict) == 2)
        loaded_items_id_list = list(interface_dict)
        self.assertIn('tt0112281', loaded_items_id_list)
        self.assertTrue('tt0112302', loaded_items_id_list)

        # test load_available_contents for index
        index_alg = IndexQuery({'Plot': 'tfidf'})
        index_dir = os.path.join(dir_test_files, 'complex_contents', 'index')
        interface_dict = index_alg._load_available_contents(index_dir)
        self.assertIsInstance(interface_dict, LoadedContentsIndex)


class TestPerUserCBAlgorithm(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.movies_multiple = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')
        cls.user_idx_list = test_ratings.unique_user_idx_column

    def test_fit(self):
        # Test fit with cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        user_str2int = train_ratings.user_map

        users_fit_dict = alg.fit(train_ratings, self.movies_multiple, num_cpus=1)

        # For the following user the algorithm could be fit
        self.assertIsNotNone(users_fit_dict.get(user_str2int["A000"]))
        self.assertIsNotNone(users_fit_dict.get(user_str2int["A001"]))
        self.assertIsNotNone(users_fit_dict.get(user_str2int["A002"]))
        self.assertIsNotNone(users_fit_dict.get(user_str2int["A003"]))

        # Test fit with the cbrs algorithm
        # For user A000 no items available locally, so the alg will not be fit for it
        users_fit_dict = alg.fit(train_ratings_some_missing, self.movies_multiple, num_cpus=1)

        # For user A000 the alg could not be fit, but it could for A001
        self.assertIsNone(users_fit_dict.get(user_str2int["A000"]))
        self.assertIsNotNone(users_fit_dict.get(user_str2int["A001"]))

    def test_rank(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # we must fit the algorithm in order to rank
        fit_alg = alg.fit(train_ratings, self.movies_multiple, num_cpus=1)

        # Test unbound ranking with the cbrs algorithm with testratings methodology
        result_rank_filtered = alg.rank(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                        user_idx_list=self.user_idx_list, n_recs=None,
                                        methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                        num_cpus=1)

        # assert that for each user the length of its rank is the same of its filter list
        for rank_user_uir in result_rank_filtered:
            user_idx = rank_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(rank_user_uir))

        # Test top-2 ranking with the cbrs algorithm for only some users
        # (all items methodology since the test set of a user could have less than 2 items to rank)
        top_n = 2
        cut_user_idx_list = train_ratings.user_map[["A000", "A003"]]
        result_rank_numbered = alg.rank(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                        user_idx_list=cut_user_idx_list, n_recs=top_n,
                                        methodology=AllItemsMethodology().setup(train_ratings, test_ratings),
                                        num_cpus=1)

        # assert that we get a rank only for the users we specified
        self.assertEqual(set(cut_user_idx_list), set(np.vstack(result_rank_numbered)[:, 0]))

        # assert that for each user specified, we get top-2 ranking
        for rank_user_uir in result_rank_numbered:
            self.assertTrue(len(rank_user_uir) == top_n)

        # Test algorithm could not be fit for A000
        a000_idx = train_ratings_some_missing.user_map["A000"]
        fit_alg = alg.fit(train_ratings_some_missing, self.movies_multiple, num_cpus=1)
        [result_empty] = alg.rank(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                  user_idx_list={a000_idx}, n_recs=None,
                                  methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                  num_cpus=1)
        self.assertTrue(len(result_empty) == 0)

    def test_predict(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # we must fit the algorithm in order to rank
        fit_alg = alg.fit(train_ratings, self.movies_multiple, num_cpus=1)

        # Test unbound ranking with the cbrs algorithm with testratings methodology
        result_pred_filtered = alg.predict(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                           user_idx_list=self.user_idx_list,
                                           methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                           num_cpus=1)

        # assert that for each user the length of its rank is the same of its filter list
        for pred_user_uir in result_pred_filtered:
            user_idx = pred_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(pred_user_uir))

        # Test prediction of the cbrs algorithm for only some users
        cut_user_idx_list = train_ratings.user_map[["A000", "A003"]]
        result_pred_numbered = alg.predict(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                           user_idx_list=cut_user_idx_list,
                                           methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                           num_cpus=1)

        # assert that we get a rank only for the users we specified
        self.assertEqual(set(cut_user_idx_list), set(np.vstack(result_pred_numbered)[:, 0]))

        # Test algorithm could not be fit for A000
        a000_idx = train_ratings_some_missing.user_map["A000"]
        fit_alg = alg.fit(train_ratings_some_missing, self.movies_multiple, num_cpus=1)
        [result_empty] = alg.predict(fit_alg, train_ratings, test_ratings, self.movies_multiple,
                                     user_idx_list={a000_idx},
                                     methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                     num_cpus=1)
        self.assertTrue(len(result_empty) == 0)

    def test_fit_rank_save_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # Test unbound ranking with the cbrs algorithm with testratings methodology
        fit_alg, result_rank_filtered = alg.fit_rank(train_ratings, test_ratings, self.movies_multiple,
                                                     user_idx_list=self.user_idx_list, n_recs=None,
                                                     methodology=TestRatingsMethodology().setup(train_ratings,
                                                                                                test_ratings),
                                                     num_cpus=1,
                                                     save_fit=True)

        # assert that for each user the length of its rank is the same of its filter list
        for rank_user_uir in result_rank_filtered:
            user_idx = rank_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(rank_user_uir))

        # save_fit == True, so we check algorithm fit for each user
        self.assertTrue(len(fit_alg) != 0)
        for user_idx in test_ratings.unique_user_idx_column:
            self.assertIsNotNone(fit_alg.get(user_idx))

        # Test top-2 ranking with the cbrs algorithm for only some users
        # (all items methodology since the test set of a user could have less than 2 items to rank)
        top_n = 2
        cut_user_idx_list = train_ratings.user_map[["A000", "A003"]]
        fit_alg, result_rank_numbered = alg.fit_rank(train_ratings, test_ratings, self.movies_multiple,
                                                     user_idx_list=cut_user_idx_list, n_recs=top_n,
                                                     methodology=AllItemsMethodology().setup(train_ratings,
                                                                                             test_ratings),
                                                     num_cpus=1,
                                                     save_fit=True)

        # assert that we get a rank only for the users we specified
        self.assertEqual(set(cut_user_idx_list), set(np.vstack(result_rank_numbered)[:, 0]))

        # assert that for each user specified, we get top-2 ranking
        for rank_user_uir in result_rank_numbered:
            self.assertTrue(len(rank_user_uir) == top_n)

        # save_fit == True, so we check whole algorithm is fit
        self.assertTrue(len(fit_alg) != 0)
        for user_idx in cut_user_idx_list:
            self.assertIsNotNone(fit_alg.get(user_idx))

        # check that only A000 and A003 were fit
        self.assertEqual(set(cut_user_idx_list), set(fit_alg.keys()))

        # Test algorithm not fit
        a000_idx = train_ratings_some_missing.user_map["A000"]
        fit_alg, [result_empty] = alg.fit_rank(train_ratings_some_missing, test_ratings, self.movies_multiple,
                                               user_idx_list={a000_idx}, n_recs=None,
                                               methodology=TestRatingsMethodology().setup(train_ratings, test_ratings),
                                               num_cpus=1,
                                               save_fit=True)
        self.assertTrue(len(result_empty) == 0)

        # if alg could not be fit for any selected user, it will be an empty dict
        self.assertTrue(len(fit_alg) == 0)

    def test_fit_rank_not_save_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # Test unbound ranking with the cbrs algorithm with testratings methodology
        fit_alg, result_rank_filtered = alg.fit_rank(train_ratings, test_ratings, self.movies_multiple,
                                                     user_idx_list=self.user_idx_list, n_recs=None,
                                                     methodology=TestRatingsMethodology().setup(train_ratings,
                                                                                                test_ratings),
                                                     num_cpus=1,
                                                     save_fit=False)

        # assert that for each user the length of its rank is the same of its filter list
        for rank_user_uir in result_rank_filtered:
            user_idx = rank_user_uir[0][0]  # the idx for the uir rank is in the first column first cell ([0][0])
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(rank_user_uir))

        # save_fit == False, so we check algorithm not fit
        self.assertIsNone(fit_alg)

    def test_fit_predict_save_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # Test predicting with the cbrs algorithm with testrating
        fit_alg, result_predict_filtered = alg.fit_predict(train_ratings, test_ratings, self.movies_multiple,
                                                           user_idx_list=self.user_idx_list,
                                                           methodology=TestRatingsMethodology().setup(train_ratings,
                                                                                                      test_ratings),
                                                           num_cpus=1,
                                                           save_fit=True)

        # assert that for each user the length of its predictions is the same of its filter list
        for predict_user_uir in result_predict_filtered:
            user_idx = predict_user_uir[0][0]  # the idx for the uir  is in the first column first cell ([0][0])
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(predict_user_uir))

        # save_fit == True, so we check algorithm fit for each user
        self.assertTrue(len(fit_alg) != 0)
        for user_idx in test_ratings.unique_user_idx_column:
            self.assertIsNotNone(fit_alg.get(user_idx))

        # Test predict with the cbrs algorithm for only some users
        cut_user_idx_list = train_ratings.user_map[["A000", "A003"]]
        fit_alg, result_predict_subset = alg.fit_predict(train_ratings, test_ratings, self.movies_multiple,
                                                         user_idx_list=cut_user_idx_list,
                                                         methodology=AllItemsMethodology().setup(train_ratings,
                                                                                                 test_ratings),
                                                         num_cpus=1,
                                                         save_fit=True)

        # assert that we get a score prediction only for the users we specified
        self.assertEqual(set(cut_user_idx_list), set(np.vstack(result_predict_subset)[:, 0]))

        # save_fit == True, so we check whole algorithm is fit
        self.assertTrue(len(fit_alg) != 0)
        for user_idx in cut_user_idx_list:
            self.assertIsNotNone(fit_alg.get(user_idx))

        # check that only A000 and A003 were fit
        self.assertEqual(set(cut_user_idx_list), set(fit_alg.keys()))

        # Test algorithm not fit
        a000_idx = train_ratings_some_missing.user_map["A000"]
        fit_alg, [result_empty] = alg.fit_predict(train_ratings_some_missing, test_ratings, self.movies_multiple,
                                                  user_idx_list={a000_idx},
                                                  methodology=TestRatingsMethodology().setup(train_ratings,
                                                                                             test_ratings),
                                                  num_cpus=1,
                                                  save_fit=True)
        self.assertTrue(len(result_empty) == 0)

        # if alg could not be fit for any selected user, it will be an empty dict
        self.assertTrue(len(fit_alg) == 0)

    def test_fit_predict_not_save_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())

        # Test predicting with the cbrs algorithm with testratings methodology
        fit_alg, result_predict_filtered = alg.fit_predict(train_ratings, test_ratings, self.movies_multiple,
                                                           user_idx_list=self.user_idx_list,
                                                           methodology=TestRatingsMethodology().setup(train_ratings,
                                                                                                      test_ratings),
                                                           num_cpus=1,
                                                           save_fit=False)

        # assert that for each user the length of its predictions is the same of its filter list
        for predict_user_uir in result_predict_filtered:
            user_idx = predict_user_uir[0][0]  # the idx for the uir prediction is in the first column first cell
            self.assertEqual(len(test_ratings.get_user_interactions(user_idx)), len(predict_user_uir))

        # save_fit == False, so we check algorithm not fit
        self.assertIsNone(fit_alg)


if __name__ == "__main__":
    unittest.main()

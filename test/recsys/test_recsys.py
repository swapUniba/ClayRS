import os
from unittest import TestCase

import numpy as np
import pandas as pd

from clayrs.content_analyzer import Ratings
from clayrs.recsys import LinearPredictor, SkLinearRegression, TrainingItemsMethodology, AllItemsMethodology
from clayrs.recsys.content_based_algorithm.classifier.classifiers import SkSVC
from clayrs.recsys.recsys import GraphBasedRS, ContentBasedRS
from clayrs.recsys.content_based_algorithm.classifier.classifier_recommender import ClassifierRecommender
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg, NotFittedAlg
from clayrs.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from clayrs.recsys.graphs import NXFullGraph

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
user_map = {}
users_train = train_ratings["from_id"]
users_test = test_ratings["from_id"]
users_train_some_missing = train_ratings_some_missing["from_id"]

all_users = users_train.append(users_test).append(users_train_some_missing)
for user_id in all_users:
    if user_id not in user_map:
        user_map[user_id] = len(user_map)


item_map = {}

items_train = train_ratings["to_id"]
items_test = test_ratings["to_id"]
items_train_some_missing = train_ratings_some_missing["to_id"]

all_items = items_train.append(items_test).append(items_train_some_missing)
for item_id in all_items:
    if item_id not in item_map:
        item_map[item_id] = len(item_map)


train_ratings = Ratings.from_dataframe(train_ratings, user_map=user_map, item_map=item_map)
train_ratings_some_missing = Ratings.from_dataframe(train_ratings_some_missing, user_map=user_map, item_map=item_map)
test_ratings = Ratings.from_dataframe(test_ratings, user_map=user_map, item_map=item_map)


# Each of the cbrs algorithm has its own class tests, so we just take
# one cbrs alg as example. The behaviour is the same for all cbrs alg
class TestContentBasedRS(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.movies_multiple = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

    def test_fit(self):
        # Test fit with cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        cbrs.fit(num_cpus=1)

        # we just check that the call to the cb algorithm is successful at the recsys level
        # further tests can be found for each cb algorithm
        self.assertIsNotNone(cbrs.fit_alg)

    def test_raise_error_without_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        with self.assertRaises(NotFittedAlg):
            cbrs.rank(test_ratings, num_cpus=1)

        with self.assertRaises(NotFittedAlg):
            cbrs.predict(test_ratings, num_cpus=1)

    def test_rank(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # we must fit the algorithm in order to rank
        cbrs.fit(num_cpus=1)

        # Test ranking with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_rank_all = cbrs.rank(test_ratings, methodology=None, num_cpus=1)
        self.assertTrue(len(result_rank_all) != 0)

        methodology_effectively_used = AllItemsMethodology().setup(train_ratings, test_ratings)
        for user_idx in result_rank_all.unique_user_idx_column:
            single_uir_rank = result_rank_all.get_user_interactions(user_idx)
            items_ranked = set(single_uir_rank[:, 1])

            expected_ranked_items = set(methodology_effectively_used.filter_single(user_idx,
                                                                                   train_ratings,
                                                                                   test_ratings))

            self.assertEqual(expected_ranked_items, items_ranked)

        # test ranking with the cbrs algorithm on unspecified user list
        # all users of the test set will be used
        result_rank_all = cbrs.rank(test_ratings, num_cpus=1)
        self.assertTrue(len(result_rank_all) != 0)

        np.testing.assert_array_equal(test_ratings.user_idx_column, result_rank_all.user_idx_column)

        # test ranking with the cbrs algorithm on specified STRING user list
        result_rank_str_specified = cbrs.rank(test_ratings, num_cpus=1, user_list=["A000", "A003"])
        self.assertTrue(len(result_rank_all) != 0)

        self.assertEqual(["A000", "A003"], list(result_rank_str_specified.unique_user_id_column))

        # test ranking with the cbrs algorithm on specified INT user list
        user_idx_list = test_ratings.user_map[["A000", "A003"]]
        result_rank_int_specified = cbrs.rank(test_ratings, num_cpus=1, user_list=user_idx_list)
        self.assertTrue(len(result_rank_all) != 0)

        np.testing.assert_array_equal(user_idx_list, result_rank_int_specified.unique_user_idx_column)

        # covering the case in which no item is recommended for any user
        result_rank_all = cbrs.rank(test_ratings,
                                    methodology=AllItemsMethodology(["not_existing", "not_existing2"]), num_cpus=1)
        self.assertTrue(len(result_rank_all) == 0)

    def test_predict(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # we must fit the algorithm in order to predict
        cbrs.fit(num_cpus=1)

        # Test predict with the cbrs algorithm on specified items
        result_predict_filtered = cbrs.predict(test_ratings, num_cpus=1)
        self.assertEqual(len(result_predict_filtered), len(test_ratings))

        # Test predict with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_predict_all = cbrs.predict(test_ratings, methodology=None, num_cpus=1)
        self.assertTrue(len(result_predict_all) != 0)

        # test predict with the cbrs algorithm on unspecified user list
        # all users of the test set will be used
        result_predict_all = cbrs.predict(test_ratings, num_cpus=1)
        self.assertTrue(len(result_predict_all) != 0)

        np.testing.assert_array_equal(test_ratings.user_idx_column, result_predict_all.user_idx_column)

        # test predict with the cbrs algorithm on specified STRING user list
        result_predict_str_specified = cbrs.predict(test_ratings, num_cpus=1, user_list=["A000", "A003"])
        self.assertTrue(len(result_predict_all) != 0)

        self.assertEqual(["A000", "A003"], list(result_predict_str_specified.unique_user_id_column))

        # test predict with the cbrs algorithm on specified INT user list
        user_idx_list = test_ratings.user_map[["A000", "A003"]]
        result_predict_int_specified = cbrs.predict(test_ratings, num_cpus=1, user_list=user_idx_list)
        self.assertTrue(len(result_predict_all) != 0)

        np.testing.assert_array_equal(user_idx_list, result_predict_int_specified.unique_user_idx_column)

        # covering the case in which no item is recommended for any user
        result_predict_all = cbrs.predict(test_ratings,
                                          methodology=AllItemsMethodology(["not_existing", "not_existing2"]), num_cpus=1)
        self.assertTrue(len(result_predict_all) == 0)

    def test_predict_raise_error(self):
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # You must fit first in order to predict
        cbrs.fit(num_cpus=1)

        # This will raise error since page rank is not a prediction algorithm
        with self.assertRaises(NotPredictionAlg):
            cbrs.predict(test_ratings)

    def test_fit_rank_save_fit(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # Test ranking with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_rank_all = cbrs.fit_rank(test_ratings, methodology=None, num_cpus=1, save_fit=True)
        self.assertTrue(len(result_rank_all) != 0)

        methodology_effectively_used = AllItemsMethodology().setup(train_ratings, test_ratings)
        for user_idx in result_rank_all.unique_user_idx_column:
            single_uir_rank = result_rank_all.get_user_interactions(user_idx)
            items_ranked = set(single_uir_rank[:, 1])

            expected_ranked_items = set(methodology_effectively_used.filter_single(user_idx,
                                                                                   train_ratings,
                                                                                   test_ratings))

            self.assertEqual(expected_ranked_items, items_ranked)
        
        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)
        
        # test ranking with the cbrs algorithm on unspecified user list
        # all users of the test set will be used
        result_rank_all = cbrs.rank(test_ratings, num_cpus=1)
        self.assertTrue(len(result_rank_all) != 0)

        np.testing.assert_array_equal(test_ratings.user_idx_column, result_rank_all.user_idx_column)

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # test ranking with the cbrs algorithm on specified STRING user list
        result_rank_str_specified = cbrs.rank(test_ratings, num_cpus=1, user_list=["A000", "A003"])
        self.assertTrue(len(result_rank_all) != 0)

        self.assertEqual(["A000", "A003"], list(result_rank_str_specified.unique_user_id_column))

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # test ranking with the cbrs algorithm on specified INT user list
        user_idx_list = test_ratings.user_map[["A000", "A003"]]
        result_rank_int_specified = cbrs.rank(test_ratings, num_cpus=1, user_list=user_idx_list)
        self.assertTrue(len(result_rank_all) != 0)

        np.testing.assert_array_equal(user_idx_list, result_rank_int_specified.unique_user_idx_column)

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # covering the case in which no item is recommended for any user
        result_rank_all = cbrs.fit_rank(test_ratings,
                                        methodology=AllItemsMethodology(["not_existing", "not_existing2"]), num_cpus=1,
                                        save_fit=True)
        self.assertTrue(len(result_rank_all) == 0)

    def test_fit_rank_not_save_fit(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # Test ranking with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_rank_all = cbrs.fit_rank(test_ratings, methodology=None, num_cpus=1, save_fit=True)
        self.assertTrue(len(result_rank_all) != 0)

        # save_fit == False, so check that algorithm is NOT fit
        self.assertIsNotNone(cbrs.fit_alg)

    def test_fit_predict_save_fit(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # Test predict with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_predict_all = cbrs.fit_predict(test_ratings, methodology=None, num_cpus=1, save_fit=True)
        self.assertTrue(len(result_predict_all) != 0)

        methodology_effectively_used = AllItemsMethodology().setup(train_ratings, test_ratings)
        for user_idx in result_predict_all.unique_user_idx_column:
            single_uir_rank = result_predict_all.get_user_interactions(user_idx)
            items_ranked = set(single_uir_rank[:, 1])

            expected_ranked_items = set(methodology_effectively_used.filter_single(user_idx,
                                                                                   train_ratings,
                                                                                   test_ratings))

            self.assertEqual(expected_ranked_items, items_ranked)

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # test predict with the cbrs algorithm on unspecified user list
        # all users of the test set will be used
        result_predict_all = cbrs.predict(test_ratings, num_cpus=1)
        self.assertTrue(len(result_predict_all) != 0)

        np.testing.assert_array_equal(test_ratings.user_idx_column, result_predict_all.user_idx_column)

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # test predict with the cbrs algorithm on specified STRING user list
        result_predict_str_specified = cbrs.predict(test_ratings, num_cpus=1, user_list=["A000", "A003"])
        self.assertTrue(len(result_predict_all) != 0)

        self.assertEqual(["A000", "A003"], list(result_predict_str_specified.unique_user_id_column))

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # test predict with the cbrs algorithm on specified INT user list
        user_idx_list = test_ratings.user_map[["A000", "A003"]]
        result_predict_int_specified = cbrs.predict(test_ratings, num_cpus=1, user_list=user_idx_list)
        self.assertTrue(len(result_predict_all) != 0)

        np.testing.assert_array_equal(user_idx_list, result_predict_int_specified.unique_user_idx_column)

        # save_fit == True, so check that algorithm is fit
        self.assertIsNotNone(cbrs.fit_alg)

        # covering the case in which no item is recommended for any user
        result_predict_all = cbrs.fit_predict(test_ratings,
                                              methodology=AllItemsMethodology(["not_existing", "not_existing2"]),
                                              num_cpus=1, save_fit=True)
        self.assertTrue(len(result_predict_all) == 0)

    def test_fit_predict_not_save_fit(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # Test predict with the cbrs algorithm with None methodology
        # in this case AllItemsMethodology() will be used
        result_predict_all = cbrs.fit_predict(test_ratings, methodology=None, num_cpus=1, save_fit=True)
        self.assertTrue(len(result_predict_all) != 0)

        # save_fit == False, so check that algorithm is NOT fit
        self.assertIsNotNone(cbrs.fit_alg)


class TestGraphBasedRS(TestCase):

    def setUp(self) -> None:
        # different test ratings from the cbrs since a graph based algorithm
        # can give predictions only to items that are in the graph
        test_ratings = pd.DataFrame.from_records([
            ("A000", "tt0112896", None),
            ("A000", "tt0112453", None),
            ("A001", "tt0114576", None),
            ("A001", "tt0113497", None),
            ("A002", "tt0114576", None),
            ("A002", "tt0113041", None),
            ("A003", "tt0114576", None)],
            columns=["from_id", "to_id", "score"])
        self.test_ratings = Ratings.from_dataframe(test_ratings)

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
        train_ratings = Ratings.from_dataframe(train_ratings)

        self.graph = NXFullGraph(train_ratings)

    def test_rank(self):
        # Test rank with the graph based algorithm
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)
        user_str2int = self.test_ratings.user_map

        # Test ranking with the graph based algorithm on specified items
        result_rank_filtered = gbrs.rank(self.test_ratings, num_cpus=1)
        self.assertEqual(len(result_rank_filtered), len(self.test_ratings))

        # Test ranking with the gbrs algorithm on all unseen items that are in the graph
        result_rank_all = gbrs.rank(self.test_ratings, methodology=None, num_cpus=1)
        self.assertTrue(len(result_rank_all) != 0)

        # Test top-n ranking with the gbrs algorithm only for some users
        result_rank_numbered = gbrs.rank(self.test_ratings, n_recs=2, methodology=None, user_list=["A000", "A003"],
                                         num_cpus=1)
        self.assertEqual(set(result_rank_numbered.user_id_column), {"A000", "A003"})
        for user in {"A000", "A003"}:
            result_single = result_rank_numbered.get_user_interactions(user_str2int[user])
            self.assertTrue(len(result_single) == 2)

        # Test ranking with alternative methodology
        train_ratings = self.graph.to_ratings(self.test_ratings.user_map, self.test_ratings.item_map)
        methodology_effectively_used = TrainingItemsMethodology().setup(train_ratings, self.test_ratings)
        result_different_meth = gbrs.rank(self.test_ratings, methodology=methodology_effectively_used, num_cpus=1)
        for user_idx in self.test_ratings.unique_user_idx_column:

            single_uir_rank = result_different_meth.get_user_interactions(user_idx)
            items_ranked = set(single_uir_rank[:, 1])

            expected_ranked_items = set(methodology_effectively_used.filter_single(user_idx,
                                                                                   train_ratings,
                                                                                   self.test_ratings))

            self.assertEqual(expected_ranked_items, items_ranked)

    def test_predict_raise_error(self):
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)

        # This will raise error since page rank is not a prediction algorithm
        with self.assertRaises(NotPredictionAlg):
            gbrs.predict(self.test_ratings, num_cpus=1)

    def test_rank_filterlist_empty_A000(self):
        # no items to recommend is in the graph for user A000
        test_ratings = pd.DataFrame.from_records([
            ("A000", "not_in_graph", None),
            ("A000", "not_in_graph1", None),
            ("A001", "tt0114576", None),
            ("A001", "tt0113497", None),
            ("A002", "tt0114576", None),
            ("A002", "tt0113041", None),
            ("A003", "tt0114576", None)],
            columns=["from_id", "to_id", "score"])
        self.test_ratings.item_map.append(["not_in_graph", "not_in_graph1"])
        test_ratings = Ratings.from_dataframe(test_ratings,
                                              user_map=self.test_ratings.user_map,
                                              item_map=self.test_ratings.item_map)
        user_str2int = test_ratings.user_map

        # Test rank with the graph based algorithm
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)

        # Test ranking with the graph based algorithm with items not present in the graph for A000
        result_rank = gbrs.rank(test_ratings, num_cpus=1)
        self.assertTrue(len(result_rank) != 0)

        # no rank is present for A000
        user_rank = result_rank.get_user_interactions(user_str2int['A000'])
        self.assertTrue(len(user_rank) == 0)

    def test_rank_filterlist_empty_all(self):
        # different test ratings from the cbrs since a graph based algorithm
        # can give predictions only to items that are in the graph
        test_ratings = pd.DataFrame.from_records([
            ("A000", "not_in_graph", None),
            ("A000", "not_in_graph1", None),
            ("A001", "not_in_graph2", None),
            ("A001", "not_in_graph3", None),
            ("A002", "not_in_graph4", None),
            ("A002", "not_in_graph5", None),
            ("A003", "not_in_graph6", None)],
            columns=["from_id", "to_id", "score"])
        self.test_ratings.item_map.append(["not_in_graph", "not_in_graph1", "not_in_graph2", "not_in_graph3",
                                           "not_in_graph4", "not_in_graph5", "not_in_graph6"])
        test_ratings = Ratings.from_dataframe(test_ratings,
                                              user_map=self.test_ratings.user_map,
                                              item_map=self.test_ratings.item_map)

        # Test rank with the graph based algorithm
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)

        # Test ranking with the graph based algorithm on items not in the graph, we expect it to be empty
        result_rank_empty = gbrs.rank(test_ratings, num_cpus=1)
        self.assertTrue(len(result_rank_empty) == 0)

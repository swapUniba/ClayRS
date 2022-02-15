import os
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.recsys import LinearPredictor, SkLinearRegression, TrainingItemsMethodology
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import SkSVC
from orange_cb_recsys.recsys.recsys import GraphBasedRS, ContentBasedRS
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifier_recommender import ClassifierRecommender
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg, NotFittedAlg
from orange_cb_recsys.recsys.graph_based_algorithm.page_rank.nx_page_rank import NXPageRank
from orange_cb_recsys.recsys.graphs import NXFullGraph

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
train_ratings = Ratings.from_dataframe(train_ratings)

# No locally available items for A000
train_ratings_some_missing = pd.DataFrame.from_records([
    ("A000", "not_existent1", 5, "54654675"),
    ("A001", "tt0114576", 3, "54654675"),
    ("A001", "tt0112896", 1, "54654675"),
    ("A000", "not_existent2", 5, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])
train_ratings_some_missing = Ratings.from_dataframe(train_ratings_some_missing)

test_ratings = pd.DataFrame.from_records([
    ("A000", "tt0114388", None),
    ("A000", "tt0112302", None),
    ("A001", "tt0113189", None),
    ("A001", "tt0113228", None),
    ("A002", "tt0114319", None),
    ("A002", "tt0114709", None),
    ("A003", "tt0114885", None)],
    columns=["from_id", "to_id", "score"])
test_ratings = Ratings.from_dataframe(test_ratings)


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

        cbrs.fit()

        # For the following user the algorithm could be fit
        self.assertIsNotNone(cbrs._user_fit_dic.get("A000"))
        self.assertIsNotNone(cbrs._user_fit_dic.get("A001"))
        self.assertIsNotNone(cbrs._user_fit_dic.get("A002"))
        self.assertIsNotNone(cbrs._user_fit_dic.get("A003"))

        # Test fit with the cbrs algorithm
        # For user A000 no items available locally, so the alg will not be fit for it
        cbrs_missing = ContentBasedRS(alg, train_ratings_some_missing, self.movies_multiple)

        cbrs_missing.fit()

        # For user A000 the alg could not be fit, but it could for A001
        self.assertIsNone(cbrs_missing._user_fit_dic.get("A000"))
        self.assertIsNotNone(cbrs_missing._user_fit_dic.get("A001"))

    def test_raise_error_without_fit(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        with self.assertRaises(NotFittedAlg):
            cbrs.rank(train_ratings)

        with self.assertRaises(NotFittedAlg):
            cbrs.predict(train_ratings)

    def test_rank(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # we must fit the algorithm in order to rank
        cbrs.fit()

        # Test ranking with the cbrs algorithm on specified items
        result_rank_filtered = cbrs.rank(test_ratings)
        self.assertEqual(len(result_rank_filtered), len(test_ratings))

        # Test ranking with the cbrs algorithm on all available unseen items
        result_rank_all = cbrs.rank(test_ratings, methodology=None)
        self.assertTrue(len(result_rank_all) != 0)

        # Test top-n ranking with the cbrs algorithm
        result_rank_numbered = cbrs.rank(test_ratings, n_recs=2, methodology=None)
        for user in set(test_ratings.user_id_column):
            result_single = result_rank_numbered.get_user_interactions(user)
            self.assertTrue(len(result_single) == 2)

        # Test ranking with alternative methodology
        result_different_meth = cbrs.rank(test_ratings, methodology=TrainingItemsMethodology())
        for user in set(test_ratings.user_id_column):
            result_single = result_different_meth.get_user_interactions(user)
            result_single_items = set([result_interaction.item_id for result_interaction in result_single])
            items_already_seen_user = set([train_interaction.item_id
                                           for train_interaction in train_ratings.get_user_interactions(user)])
            items_expected_rank = set([train_interaction.item_id
                                       for train_interaction in train_ratings
                                       if train_interaction.item_id not in items_already_seen_user])

            self.assertEqual(items_expected_rank, result_single_items)

        # Test algorithm not fitted
        cbrs = ContentBasedRS(alg, train_ratings_some_missing, self.movies_multiple)

        cbrs.fit()
        result_empty = cbrs.rank(test_ratings, user_id_list=['A000'])
        self.assertTrue(len(result_empty) == 0)

    def test_predict(self):
        # Test fit with the cbrs algorithm
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # we must fit the algorithm in order to predict
        cbrs.fit()

        # Test predict with the cbrs algorithm on specified items
        result_predict_filtered = cbrs.predict(test_ratings)
        self.assertEqual(len(result_predict_filtered), len(test_ratings))

        # Test predict with the cbrs algorithm on all available unseen items
        result_predict_all = cbrs.predict(test_ratings, methodology=None)
        self.assertTrue(len(result_predict_all) != 0)

        # Test predict with alternative methodology
        result_different_meth = cbrs.predict(test_ratings, methodology=TrainingItemsMethodology())
        for user in set(test_ratings.user_id_column):
            result_single = result_different_meth.get_user_interactions(user)
            result_single_items = set([result_interaction.item_id for result_interaction in result_single])
            items_already_seen_user = set([train_interaction.item_id
                                           for train_interaction in train_ratings.get_user_interactions(user)])
            items_expected_rank = set([train_interaction.item_id
                                       for train_interaction in train_ratings
                                       if train_interaction.item_id not in items_already_seen_user])

            self.assertEqual(items_expected_rank, result_single_items)

        # Test algorithm not fitted
        cbrs = ContentBasedRS(alg, train_ratings_some_missing, self.movies_multiple)

        cbrs.fit()
        result_empty = cbrs.rank(test_ratings, user_id_list=['A000'])
        self.assertTrue(len(result_empty) == 0)

    def test_predict_raise_error(self):
        alg = ClassifierRecommender({'Plot': ['tfidf', 'embedding']}, SkSVC())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        # You must fit first in order to predict
        cbrs.fit()

        # This will raise error since page rank is not a prediction algorithm
        with self.assertRaises(NotPredictionAlg):
            cbrs.predict(test_ratings)

    def test_fit_rank(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        result = cbrs.fit_rank(test_ratings)

        # No further test since the fit_rank() method just calls the fit() method and rank() method
        self.assertTrue(len(result) != 0)

    def test_fit_predict(self):
        alg = LinearPredictor({'Plot': ['tfidf', 'embedding']}, SkLinearRegression())
        cbrs = ContentBasedRS(alg, train_ratings, self.movies_multiple)

        result = cbrs.fit_predict(test_ratings)

        # No further test since the fit_predict() method just calls the fit() method and rank() method
        self.assertTrue(len(result) != 0)


class TestGraphBasedRS(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # different test ratings from the cbrs since a graph based algorithm
        # can give predictions only to items that are in the graph
        cls.test_ratings = pd.DataFrame.from_records([
            ("A000", "tt0112896"),
            ("A000", "tt0112453"),
            ("A001", "tt0114576"),
            ("A001", "tt0113497"),
            ("A002", "tt0114576"),
            ("A002", "tt0113041"),
            ("A003", "tt0114576")],
            columns=["from_id", "to_id"])

        cls.graph = NXFullGraph(train_ratings)

    def test_rank(self):
        # Test rank with the graph based algorithm
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)

        # Test ranking with the graph based algorithm on specified items
        result_rank_filtered = gbrs.rank(self.test_ratings)
        self.assertEqual(len(result_rank_filtered), len(self.test_ratings))

        # Test ranking with the gbrs algorithm on all unseen items that are in the graph
        result_rank_all = gbrs.rank(test_ratings['from_id'])
        self.assertTrue(len(result_rank_all) != 0)

        # Test top-n ranking with the gbrs algorithm
        result_rank_numbered = gbrs.rank(test_ratings['from_id'], n_recs=2)
        for user in set(test_ratings['from_id']):
            result_single = result_rank_numbered[result_rank_numbered['from_id'] == user]
            self.assertTrue(len(result_single) == 2)

        # Test ranking with alternative methodology
        result_different_meth = gbrs.rank(test_ratings, methodology=TrainingItemsMethodology())
        for user in set(test_ratings['from_id']):
            result_single = result_different_meth[result_different_meth['from_id'] == user]
            items_already_seen_user = set(train_ratings[train_ratings['from_id'] == user]['to_id'])
            items_expected_rank = train_ratings[(~train_ratings['to_id'].isin(items_already_seen_user))]

            self.assertTrue(set(items_expected_rank['to_id']), set(result_single['to_id']))

    def test_predict_raise_error(self):
        alg = NXPageRank()
        gbrs = GraphBasedRS(alg, self.graph)

        # This will raise error since page rank is not a prediction algorithm
        with self.assertRaises(NotPredictionAlg):
            gbrs.predict(self.test_ratings)

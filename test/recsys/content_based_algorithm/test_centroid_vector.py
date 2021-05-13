import lzma
import os
import pickle
from unittest import TestCase

from orange_cb_recsys.recsys import RecSysConfig, RecSys

import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.content_based_algorithm.similarities import CosineSimilarity
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')


class TestCentroidVector(TestCase):

    def setUp(self) -> None:
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", -0.2, "54654675"),
            ("A001", "tt0114576", 0.8, "54654675"),
            ("A001", "tt0112896", -0.4, "54654675"),
            ("A000", "tt0113041", 0.6, "54654675"),
            ("A002", "tt0112453", -0.2, "54654675"),
            ("A002", "tt0113497", 0.5, "54654675"),
            ("A003", "tt0112453", -0.8, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        self.filter_list = ['tt0112641', 'tt0112760', 'tt0112896']

        self.movies_dir = os.path.join(contents_path, 'movies_multiple_repr/')

    def test_predict(self):

        alg = CentroidVector({'Plot': '0'}, CosineSimilarity(), 0)
        alg.initialize(self.ratings, self.movies_dir)

        pred_result = alg.fit_predict('A000', self.filter_list)
        self.assertEqual(len(pred_result), len(self.filter_list))

        rank_result = alg.fit_rank('A000', recs_number=5)
        self.assertEqual(len(rank_result), 5)

        alg = CentroidVector({"Plot": ["0", "1"],
                              "Genre": ["0", "1"],
                              "Director": "1"},
                             CosineSimilarity(),
                             0)
        alg.initialize(self.ratings, self.movies_dir)

        pred_result = alg.fit_predict('A000', self.filter_list)
        self.assertEqual(len(pred_result), len(self.filter_list))

        rank_result = alg.fit_rank('A000', recs_number=5)
        self.assertEqual(len(rank_result), 5)

    def test_empty_frame(self):

        # Only negative available
        self.ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", -1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = CentroidVector({'Plot': '0'}, CosineSimilarity(), 0)
        alg.initialize(self.ratings, self.movies_dir)

        result = alg.fit_predict('A000')
        self.assertTrue(result.empty)

        # Non Existent Item avilable locally
        self.ratings = pd.DataFrame.from_records([
            ("A000", "non existent", 0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        alg = CentroidVector({'Plot': '0'}, CosineSimilarity(), 0)
        alg.initialize(self.ratings, self.movies_dir)

        result = alg.fit_predict('A000')
        self.assertTrue(result.empty)

    def test_ino(self):
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 1, "54654675"),
            ("A000", "tt0112453", 1, "54654675"),
            ("A000", "tt0112896", -1, "54654675"),
            ("A000", "tt0113041", -1, "54654675"),
            ("A000", "tt0113497", -1, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        candidate_list = ratings.to_id

        alg = CentroidVector({'Plot': ['0']}, CosineSimilarity(), threshold=0)
        alg.initialize(ratings, self.movies_dir)

        print(alg.fit_predict('A000', candidate_list))
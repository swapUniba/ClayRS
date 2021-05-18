import os
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.recsys.content_based_algorithm.index_query import IndexQuery
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')


class TestIndexQuery(TestCase):

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

        self.movies_dir = os.path.join(contents_path, 'movies_multiple_repr_INDEX/')

    def test_predict(self):

        alg = IndexQuery({'Plot': '0'}, threshold=0)
        alg.initialize(self.ratings, self.movies_dir)

        # Will raise Exception since it's not a Score Prediction Algorithm
        with self.assertRaises(NotPredictionAlg):
            alg.fit_predict('A000')

    def test_rank(self):

        # Test single representation
        alg = IndexQuery({'Plot': '0'}, threshold=0)
        alg.initialize(self.ratings, self.movies_dir)

        rank_result = alg.fit_rank('A000', recs_number=5)
        self.assertEqual(len(rank_result), 5)

        # Test multiple representation
        alg = IndexQuery({'Plot': ['0', '1']}, threshold=0)
        alg.initialize(self.ratings, self.movies_dir)

        rank_result = alg.fit_rank('A000', recs_number=5)
        self.assertEqual(len(rank_result), 5)

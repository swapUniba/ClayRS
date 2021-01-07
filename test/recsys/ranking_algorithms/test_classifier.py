from unittest import TestCase

import lzma
import pandas as pd
import os
import pickle

from orange_cb_recsys.recsys.ranking_algorithms.classifier import ClassifierRecommender


class TestClassifierRecommender(TestCase):
    def test_predict(self):

        alg = ClassifierRecommender("Plot", "2", "gaussian_process", 0)
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0114576", 0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        try:
            path = "../../../contents/movielens_test1591885241.5520566"
            file = os.path.join(path, "tt0114576.xz")
            with lzma.open(file, "r") as content_file:
                pass
        except FileNotFoundError:
            path = "contents/movielens_test1591885241.5520566"

        self.assertGreater(alg.predict('A000', ratings, 1, path, ['tt0114576']).rating[0], 0)

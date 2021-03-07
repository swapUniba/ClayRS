from unittest import TestCase

import lzma
import pandas as pd
import os

from orange_cb_recsys.recsys.ranking_algorithms.classifier import ClassifierRecommender, KNN, RandomForest, \
    GaussianProcess, LogReg, DecisionTree, SVM


class TestClassifierRecommender(TestCase):
    def test_predict(self):

        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", 0.5, "54654675"),
            ("A000", "tt0112302", 0.5, "54654675"),
            ("A000", "tt0112346", -0.5, "54654675"),
            ("A000", "tt0112453", -0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        try:
            path = "../../../contents/movies_multiple_repr"
            file = os.path.join(path, "tt0112453.xz")
            with lzma.open(file, "r") as content_file:
                pass
        except FileNotFoundError:
            path = "contents/movies_multiple_repr"

        alg = ClassifierRecommender({"Plot": "1"}, GaussianProcess(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        alg = ClassifierRecommender({"Plot": "0"}, RandomForest(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        alg = ClassifierRecommender({"Plot": "0"}, LogReg(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        alg = ClassifierRecommender({"Plot": "0"}, KNN(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        alg = ClassifierRecommender({"Plot": "0"}, SVM(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        alg = ClassifierRecommender({"Plot": "0"}, DecisionTree(), 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

        # TEST WITH MULTIPLE FIELDS
        alg = ClassifierRecommender(
                                    item_field={"Plot": ["0", "1"],
                                                "Genre": ["0", "1"],
                                                "Director": ["1"]
                                                },
                                    classifier=KNN(n_neighbors=3),
                                    threshold=0,
                                    )
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreaterEqual(result.rating[0], 0)

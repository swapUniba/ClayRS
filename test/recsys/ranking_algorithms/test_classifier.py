from unittest import TestCase

import lzma
import pandas as pd
import os
import pickle

from orange_cb_recsys.recsys.ranking_algorithms.classifier import ClassifierRecommender


class TestClassifierRecommender(TestCase):
    def test_predict(self):
        ratings = pd.DataFrame.from_records([
            ("A000", "tt0029583", 0.5, "54654675"),
            ("A000", "tt0032910", 0.5, "54654675"),
            ("A000", "tt0048473", -0.5, "54654675"),
            ("A000", "tt0052572", -0.5, "54654675")],
            columns=["from_id", "to_id", "score", "timestamp"])

        try:
            path = "../../../contents/examples/ex_1/movies_1600355972.49884"
            file = os.path.join(path, "tt0052572.xz")
            with lzma.open(file, "r") as content_file:
                pass
        except FileNotFoundError:
            path = "contents/examples/ex_1/movies_1600355972.49884"

        alg = ClassifierRecommender("Plot", "0", "gaussian_process", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        alg = ClassifierRecommender("Plot", "0", "random_forest", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        alg = ClassifierRecommender("Plot", "0", "log_regr", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        alg = ClassifierRecommender("Plot", "0", "knn", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        alg = ClassifierRecommender("Plot", "0", "svm", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        alg = ClassifierRecommender("Plot", "0", "decision_tree", 0)
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

        # TEST WITH MULTIPLE FIELDS
        alg = ClassifierRecommender(classifier="knn",
                                    threshold=0,
                                    _fields_representations={"Plot": ["0"], "Year": ["0"]},
                                    _item_fields=["Plot", "Year"],
                                    classifier_parameters={"n_neighbors": 3})
        result = alg.predict('A000', ratings, 1, path, ['tt0114576'])
        self.assertGreater(result.rating[0], 0)

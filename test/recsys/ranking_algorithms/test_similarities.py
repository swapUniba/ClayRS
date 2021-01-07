from unittest import TestCase
from orange_cb_recsys.recsys.ranking_algorithms.similarities import CosineSimilarity, DenseVector


class TestCosineSimilarity(TestCase):
    def test_perform(self):
        a = [5, 9, 7, 8, 3, 5, 4, 2, 6, 4]
        b = [8, 1, 3, 10, 8, 4, 9, 2, 1, 6]
        sim = CosineSimilarity()
        self.assertEqual(sim.perform(DenseVector(a), DenseVector(b)),
                         0.7552110293516224)

    def exception_perform(self):
        a = [5, 9, 7, 8, 3, 5, 4, 2, 6, 4]
        b = [8, 1, 3, 10, 8, 4, 9, 2, 1, 6]
        sim = CosineSimilarity()
        with self.assertRaises(ValueError):
            sim.perform(a, b)


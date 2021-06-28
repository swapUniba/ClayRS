from unittest import TestCase
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
import numpy as np


class TestCosineSimilarity(TestCase):
    def test_perform(self):
        a = np.array([5, 9, 7, 8, 3, 5, 4, 2, 6, 4])
        b = np.array([8, 1, 3, 10, 8, 4, 9, 2, 1, 6])
        sim = CosineSimilarity()
        self.assertAlmostEqual(sim.perform(a, b),
                         0.7552110293516224)

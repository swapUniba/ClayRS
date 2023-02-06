from unittest import TestCase
from clayrs.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


class TestCosineSimilarity(TestCase):
    def test_perform(self):
        sim = CosineSimilarity()

        # vector comparison
        a = np.array([[5, 9, 7, 8, 3, 5, 4, 2, 6, 4]])
        b = np.array([[8, 1, 3, 10, 8, 4, 9, 2, 1, 6]])

        res = sim.perform(a, b)
        expected = cosine_similarity(a, b, dense_output=True)

        np.testing.assert_allclose(expected, res)

        # single vector vs one matrix comparison
        a = np.array([[5, 9, 7, 8, 3, 5, 4, 2, 6, 4]])
        b = np.array([[8, 1, 3, 10, 8, 4, 9, 2, 1, 6],
                      [8, 5, 5, 6, 2, 3, 10, 2, 3, 4],
                      [1, 2, 2, 4, 4, 7, 6, 5, 5, 3]])

        res = sim.perform(a, b)
        expected = cosine_similarity(a, b, dense_output=True)

        # check that we compute a similarity for each pair
        self.assertTrue(res.shape[0] == 1 and res.shape[1] == 3)

        np.testing.assert_allclose(expected, res)

        # sparse comparison
        a = sparse.csr_matrix(np.array([[5, 9, 7, 8, 3, 5, 4, 2, 6, 4]]))
        b = sparse.csr_matrix(np.array([[8, 1, 3, 10, 8, 4, 9, 2, 1, 6]]))

        res = sim.perform(a, b)
        expected = cosine_similarity(a, b, dense_output=True)

        np.testing.assert_allclose(expected, res)

        # single sparse vs sparse matrix comparison
        a = sparse.csr_matrix(np.array([[5, 9, 7, 8, 3, 5, 4, 2, 6, 4]]))
        b = sparse.csr_matrix(np.array([[8, 1, 3, 10, 8, 4, 9, 2, 1, 6],
                                        [8, 5, 5, 6, 2, 3, 10, 2, 3, 4],
                                        [1, 2, 2, 4, 4, 7, 6, 5, 5, 3]]))

        res = sim.perform(a, b)
        expected = cosine_similarity(a, b, dense_output=True)

        # check that we compute a similarity for each pair
        self.assertTrue(res.shape[0] == 1 and res.shape[1] == 3)

        np.testing.assert_allclose(expected, res)

        # single vector vs sparse matrix comparison
        a = np.array([[5, 9, 7, 8, 3, 5, 4, 2, 6, 4]])
        b = sparse.csr_matrix(np.array([[8, 1, 3, 10, 8, 4, 9, 2, 1, 6],
                                        [8, 5, 5, 6, 2, 3, 10, 2, 3, 4],
                                        [1, 2, 2, 4, 4, 7, 6, 5, 5, 3]]))

        res = sim.perform(a, b)
        expected = cosine_similarity(a, b, dense_output=True)

        # check that we compute a similarity for each pair
        self.assertTrue(res.shape[0] == 1 and res.shape[1] == 3)

        np.testing.assert_allclose(expected, res)

from unittest import TestCase
import numpy as np

from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid, Sum, SingleToken


class TestCentroid(TestCase):
    def test_combine(self):
        z = np.ndarray(shape=(3, 3))

        z[0, :] = [1, 1, 1]
        z[1, :] = [2, 2, 2]
        z[2, :] = [3, 3, 3]

        combiner = Centroid()
        result = combiner.combine(z)

        expected = np.ndarray(shape=(3, ))
        expected[:] = [2, 2, 2]

        self.assertTrue((result == expected).all())


class TestSum(TestCase):
    def test_combine(self):
        z = np.ndarray(shape=(3, 3))

        z[0, :] = [1, 9, 1]
        z[1, :] = [7, 2, 4]
        z[2, :] = [3, 5, 3]

        combiner = Sum()
        result = combiner.combine(z)

        expected = np.ndarray(shape=(3, ))
        expected[:] = [11, 16, 8]

        self.assertTrue((result == expected).all())


class TestSingleToken(TestCase):
    def test_combine(self):
        z = np.ndarray(shape=(3, 3))

        z[0, :] = [1, 9, 1]
        z[1, :] = [7, 2, 4]
        z[2, :] = [3, 5, 3]

        combiner = SingleToken(0)
        result = combiner.combine(z)

        expected = np.ndarray(shape=(3, ))
        expected[:] = [1, 9, 1]

        self.assertTrue((result == expected).all())

        combiner = SingleToken(2)
        result = combiner.combine(z)

        expected = np.ndarray(shape=(3, ))
        expected[:] = [3, 5, 3]

        self.assertTrue((result == expected).all())

    def test_raise(self):
        z = np.ndarray(shape=(3, 3))

        z[0, :] = [1, 9, 1]
        z[1, :] = [7, 2, 4]
        z[2, :] = [3, 5, 3]

        with self.assertRaises(IndexError):
            SingleToken(99).combine(z)

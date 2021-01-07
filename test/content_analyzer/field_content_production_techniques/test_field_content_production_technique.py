from unittest import TestCase
import numpy as np

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_source import \
    GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EmbeddingTechnique


class TestEmbeddingTechnique(TestCase):
    def test_produce_content(self):
        self.skipTest("SLOW")
        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="doc")

        result = technique.produce_content("Embedding", "title plot")
        expected = np.ndarray(shape=(25,))
        expected[:] = [7.88080007e-01, 2.99764998e-01, 4.93862494e-02, -2.96350002e-01,
                       3.28214996e-01, -8.11504990e-01, 1.06998003e+00, -2.28915006e-01,
                       4.35259998e-01, -4.70495000e-01, 2.06634995e-01, 7.93949991e-01,
                       -2.69545007e+00, 5.88585012e-01, 3.43510002e-01, 3.44478448e-01,
                       4.31589991e-01, 1.02359980e-01, 1.50011199e-01, -1.35000050e-03,
                       -7.03384009e-01, 6.97145015e-01, 5.35014980e-02, -8.15229982e-01,
                       -6.40249997e-01]

        self.assertTrue(np.allclose(result.value, expected))

        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="word")

        result = technique.produce_content("Embedding", "title plot")
        print(type(result))
        expected = np.ndarray(shape=(2, 25))
        expected[0, :] = np.array([8.5013e-01, 4.5262e-01, -7.0575e-03, -8.7738e-01, 4.2448e-01,
                                   -8.3659e-01, 8.0416e-01, 3.7408e-01, 4.3085e-01, -6.3936e-01,
                                   1.1939e-01, 1.1342e+00, -3.2065e+00, 9.3146e-01, 3.6542e-01,
                                   -3.1931e-03, 1.9790e-01, -3.2954e-01, 2.9672e-01, 4.8869e-01,
                                   -1.3787e+00, 7.5234e-01, 2.0334e-01, -6.7998e-01, -8.9194e-01])

        expected[1, :] = np.array([0.72603, 0.14691, 0.10583, 0.28468, 0.23195,
                                   -0.78642, 1.3358, -0.83191, 0.43967, -0.30163,
                                   0.29388, 0.4537, -2.1844, 0.24571, 0.3216,
                                   0.69215, 0.66528, 0.53426, 0.0033024, -0.49139,
                                   -0.028068, 0.64195, -0.096337, -0.95048, -0.38856])

        self.assertTrue(np.allclose(result.get_value(), expected))

        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="sentence")

        result = technique.produce_content("Embedding", "god is great! i won lottery.")
        expected = np.ndarray(shape=(2, 25))
        expected[0, :] = [-0.37618333, 0.15434666, -0.28881, -0.07587333, -0.27056333,
                          -0.09340801, 1.50040003, -0.06122, -0.62004667, -0.33889667,
                          -0.10680646, 0.85774666, -5.3046999, 0.41951333, 0.19130667,
                          -0.04783767, 0.32928667, 0.33661, -0.2041, 0.02257,
                          -0.18374199, 0.36268666, -0.64598666, 0.44400333, -0.05863967]
        expected[1, :] = [2.76493341e-01, 9.70453342e-01, 1.03293339e-01, -1.09362668e+00,
                          -5.29576662e-01, -6.43019984e-01, 6.42999992e-01, -3.99965997e-01,
                          -5.14123331e-01, -1.44426664e-01, 3.28413328e-01, 2.87829664e-01,
                          -4.20490011e+00, 2.52440006e-01, 1.76970005e-01, -2.68866668e-02,
                          5.24243347e-01, -5.96833328e-01, -3.44410002e-01, -1.23000145e-03,
                          -4.11850005e-01, 5.29021009e-01, 3.65209666e-01, -3.64499937e-02,
                          -5.01233364e-02]

        self.assertTrue(np.allclose(result.get_value(), expected))

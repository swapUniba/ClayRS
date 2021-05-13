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
        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="doc")

        result = technique.produce_content("title plot")
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

        result = technique.produce_content("title plot")

        expected = np.ndarray(shape=(2, 25))
        expected[0, :] = np.array([8.50130022e-01, 4.52620000e-01, -7.05750007e-03, -8.77380013e-01,
                                  4.24479991e-01, -8.36589992e-01, 8.04159999e-01, 3.74080002e-01,
                                  4.30849999e-01, -6.39360011e-01, 1.19390003e-01, 1.13419998e+00,
                                  -3.20650005e+00, 9.31460023e-01, 3.65420014e-01, -3.19309998e-03,
                                  1.97899997e-01, -3.29540014e-01, 2.96719998e-01, 4.88689989e-01,
                                  -1.37870002e+00, 7.52340019e-01, 2.03339994e-01, -6.79979980e-01,
                                  -8.91939998e-01])

        expected[1, :] = np.array([7.26029992e-01, 1.46909997e-01, 1.05829999e-01, 2.84680009e-01,
                                 2.31950000e-01, -7.86419988e-01, 1.33580005e+00, -8.31910014e-01,
                                 4.39669997e-01, -3.01629990e-01, 2.93879986e-01, 4.53700006e-01,
                                 -2.18440008e+00, 2.45710000e-01, 3.21599990e-01, 6.92149997e-01,
                                 6.65279984e-01, 5.34259975e-01, 3.30240000e-03, -4.91389990e-01,
                                 -2.80680005e-02, 6.41950011e-01, -9.63369980e-02, -9.50479984e-01,
                                 -3.88559997e-01])

        self.assertTrue(np.allclose(result.value, expected))

        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="sentence")

        result = technique.produce_content("god is great! i won lottery.")
        expected = np.ndarray(shape=(2, 25))

        expected[0, :] = [-3.76183331e-01, 1.54346665e-01, -2.88810000e-01, -7.58733253e-02,
                          -2.70563329e-01, -9.34080146e-02, 1.50040003e+00, -6.12200002e-02,
                          -6.20046665e-01, -3.38896669e-01, -1.06806465e-01, 8.57746661e-01,
                          -5.30469990e+00, 4.19513330e-01, 1.91306671e-01, -4.78376672e-02,
                          3.29286670e-01, 3.36609999e-01, -2.04100000e-01, 2.25700041e-02,
                          -1.83741993e-01, 3.62686664e-01, -6.45986656e-01, 4.44003331e-01,
                          -5.86396679e-02]
        expected[1, :] = [2.76493341e-01, 9.70453342e-01, 1.03293339e-01, -1.09362668e+00,
                          -5.29576662e-01, -6.43019984e-01, 6.42999992e-01, -3.99965997e-01,
                          -5.14123331e-01, -1.44426664e-01, 3.28413328e-01, 2.87829664e-01,
                          -4.20490011e+00, 2.52440006e-01, 1.76970005e-01, -2.68866668e-02,
                          5.24243347e-01, -5.96833328e-01, -3.44410002e-01, -1.23000145e-03,
                          -4.11850005e-01, 5.29021009e-01, 3.65209666e-01, -3.64499937e-02,
                          -5.01233364e-02]

        self.assertTrue(np.allclose(result.value, expected))

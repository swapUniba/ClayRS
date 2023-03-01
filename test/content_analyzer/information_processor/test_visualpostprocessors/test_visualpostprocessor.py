from unittest import TestCase

from clayrs.content_analyzer.information_processor.postprocessors import CountVisualBagOfWords, \
    TfIdfVisualBagOfWords, ScipyVQ, SkLearnPCA, SkLearnGaussianRandomProjections, SkLearnFeatureAgglomeration
from clayrs.content_analyzer.content_representation.content import EmbeddingField, FeaturesBagField

import numpy as np
import scipy.sparse as sp


class TestVisualPostProcessing(TestCase):

    def test_3dim_errors(self):
        input = [
            EmbeddingField(np.array([[[10, 10, 10]]])),
            EmbeddingField(np.array([[[9.8, 9.8, 9.8]]])),
            EmbeddingField(np.array([[[1, 1, 1]]]))
        ]

        with self.assertRaises(ValueError):
            CountVisualBagOfWords(n_clusters=2, random_state=42).process(input)

        with self.assertRaises(ValueError):
            TfIdfVisualBagOfWords(n_clusters=2, random_state=42).process(input)

        with self.assertRaises(ValueError):
            ScipyVQ(n_clusters=2, random_state=42).process(input)

        with self.assertRaises(ValueError):
            SkLearnPCA(n_components=2, random_state=42).process(input)

    def test_1dim_errors(self):
        input = [
            EmbeddingField(np.array([10, 10, 10])),
            EmbeddingField(np.array([9.8, 9.8, 9.8])),
            EmbeddingField(np.array([1, 1, 1]))
        ]

        with self.assertRaises(ValueError):
            CountVisualBagOfWords(n_clusters=2, random_state=42).process(input)

        with self.assertRaises(ValueError):
            TfIdfVisualBagOfWords(n_clusters=2, random_state=42).process(input)

    def test_visual_bag_of_words_count(self):
        input = [
            EmbeddingField(np.array([[10, 10, 10], [10, 10, 10]])),
            EmbeddingField(np.array([[9.8, 9.8, 9.8], [9.8, 9.8, 9.8]])),
            EmbeddingField(np.array([[1, 1, 1], [10, 10, 10]]))
        ]

        output = CountVisualBagOfWords(n_clusters=2, random_state=42).process(input)

        # Expectation: 2 clusters were defined, so the codewords dictionary should have 2 words
        # It is expected that 1 cluster will consider [10, 10, 10] and [9.8, 9.8, 9.8], while the second cluster
        # only of [1, 1, 1]
        # With the previously defined assumptions, it is supposed that, because of the weighting scema, the first two
        # embeddings will become [2, 0] (so closer to the first cluster as expected), while the second array
        # will be [1, 1] (the first feature closer to the second cluster and the second one to the first)

        self.assertIsInstance(output[0], FeaturesBagField)
        np.testing.assert_array_equal(output[0].value.toarray().squeeze(), np.array([2, 0]))
        np.testing.assert_array_equal(output[1].value.toarray().squeeze(), np.array([2, 0]))
        np.testing.assert_array_equal(output[2].value.toarray().squeeze(), np.array([1, 1]))

    def test_visual_bag_of_words_tfidf(self):
        input = [
            EmbeddingField(np.array([[10, 10, 10], [10, 10, 10]])),
            EmbeddingField(np.array([[9.8, 9.8, 9.8], [9.8, 9.8, 9.8]])),
            EmbeddingField(np.array([[1, 1, 1], [10, 10, 10]]))
        ]

        output = TfIdfVisualBagOfWords(n_clusters=2, norm=None, smooth_idf=False, random_state=42).process(input)

        # same as the CountVisualBagOfWords scenario, but this time the weighting schema is a tf-idf one
        # no smoothing or norm will be applied because of the parameters set in the constructor
        # the computation should be the following:
        # first vector: [2, 0] (only contains the first term 2 times so tf-idf for that term should be 2 * 1)
        # second vector: same as before
        # third vector: [1, 2.09]
        # for the last case:
        # 1 = 1 * ((log_e(3 / 3)) + 1)
        # 2.09 = 1 * ((log_e(3 / 1)) + 1)

        expected_tf_idf = 1 * ((np.log(3 / 1)) + 1)

        self.assertIsInstance(output[0], FeaturesBagField)
        np.testing.assert_array_equal(output[0].value.toarray().squeeze(), np.array([2, 0]))
        np.testing.assert_array_equal(output[1].value.toarray().squeeze(), np.array([2, 0]))
        np.testing.assert_almost_equal(output[2].value.toarray().squeeze(), np.array([1, expected_tf_idf]))

    def test_vq(self):

        # 1 Dim test
        input = [
            EmbeddingField(np.array([10, 10, 10])),
            EmbeddingField(np.array([9.8, 9.8, 9.8])),
            EmbeddingField(np.array([1, 1, 1]))
        ]

        output = ScipyVQ(n_clusters=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        np.testing.assert_array_equal(output[0].value, output[1].value)
        self.assertFalse(np.array_equal(output[0].value, output[2].value))
        self.assertFalse(np.array_equal(output[1].value, output[2].value))

        # 2 dim test
        input = [
            EmbeddingField(np.array([[10, 10, 10], [10, 10, 10]])),
            EmbeddingField(np.array([[9.8, 9.8, 9.8], [9.8, 9.8, 9.8]])),
            EmbeddingField(np.array([[1, 1, 1], [10, 10, 10]]))
        ]

        output = ScipyVQ(n_clusters=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        np.testing.assert_array_equal(output[0].value, output[1].value)
        self.assertFalse(np.array_equal(output[0].value, output[2].value))
        self.assertFalse(np.array_equal(output[1].value, output[2].value))
        np.testing.assert_array_equal(output[0].value[0], output[2].value[1])

    def test_sklearn_dimensionality_reduction(self):

        # for dimensionality reduction techniques it is just checked that the dimensions of the output
        # match the number of desired dimensions for the output (both in the 1 dimensional and 2 dimensional cases)

        # 1 dimensional case

        input = [
            EmbeddingField(np.array([10, 10, 9, 9, 10])),
            EmbeddingField(np.array([10, 9, 10, 6, 8])),
            EmbeddingField(np.array([12, 7, 10, 5, 2]))
        ]

        output = SkLearnPCA(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)

        output = SkLearnGaussianRandomProjections(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)

        output = SkLearnFeatureAgglomeration().process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)

        # 2 dimensional case

        input = [
            EmbeddingField(np.array([[10, 10, 9, 9, 10], [12, 7, 10, 5, 2]])),
            EmbeddingField(np.array([[10, 9, 10, 6, 8], [10, 10, 9, 9, 10]])),
            EmbeddingField(np.array([[12, 7, 10, 5, 2], [10, 9, 10, 6, 8]]))
        ]

        output = SkLearnPCA(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(output[0].value.shape, (2, 2))
        self.assertEqual(output[1].value.shape, (2, 2))
        self.assertEqual(output[2].value.shape, (2, 2))

        output = SkLearnGaussianRandomProjections(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(output[0].value.shape, (2, 2))
        self.assertEqual(output[1].value.shape, (2, 2))
        self.assertEqual(output[2].value.shape, (2, 2))

        output = SkLearnFeatureAgglomeration().process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(output[0].value.shape, (2, 2))
        self.assertEqual(output[1].value.shape, (2, 2))
        self.assertEqual(output[2].value.shape, (2, 2))

        # 1 dimensional sparse csc case
        # in the case of bags of features only the 1 dimensional case makes sense

        input = [
            FeaturesBagField(sp.csr_matrix(np.array([10, 10, 9, 9, 10])).tocsc(), pos_feature_tuples=[]),
            FeaturesBagField(sp.csr_matrix(np.array([10, 9, 10, 6, 8])).tocsc(), pos_feature_tuples=[]),
            FeaturesBagField(sp.csr_matrix(np.array([12, 7, 10, 5, 2])).tocsc(), pos_feature_tuples=[])
        ]

        output = SkLearnPCA(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)

        output = SkLearnGaussianRandomProjections(n_components=2, random_state=42).process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)

        output = SkLearnFeatureAgglomeration().process(input)

        self.assertIsInstance(output[0], EmbeddingField)
        self.assertEqual(len(output[0].value), 2)
        self.assertEqual(len(output[1].value), 2)
        self.assertEqual(len(output[2].value), 2)



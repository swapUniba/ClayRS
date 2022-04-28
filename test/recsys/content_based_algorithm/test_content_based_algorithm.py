import os
from copy import deepcopy
from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from clayrs.content_analyzer import Centroid
from clayrs.recsys import IndexQuery, LinearPredictor
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict, LoadedContentsIndex
from clayrs.utils.class_utils import get_all_implemented_subclasses
from clayrs.utils.load_content import load_content_instance

from clayrs.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from clayrs.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
from test import dir_test_files


class TestContentBasedAlgorithm(TestCase):

    def setUp(self) -> None:

        # ContentBasedAlgorithm is an abstract class, so we need to instantiate
        # a subclass to test its methods.
        self.alg = CentroidVector({'Plot': 'tfidf'}, CosineSimilarity(), 0)

    def test__bracket_representation(self):

        item_field = {'Plot': 'tfidf',
                      'Genre': [0],
                      'Title': [0, 'trybracket'],
                      'Director': 5}

        item_field_bracketed = {'Plot': ['tfidf'],
                                'Genre': [0],
                                'Title': [0, 'trybracket'],
                                'Director': [5]}

        result = self.alg._bracket_representation(item_field)

        self.assertEqual(item_field_bracketed, result)

    def test_extract_features_item(self):
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

        content = load_content_instance(movies_dir, 'tt0112281')

        result = self.alg.extract_features_item(content)

        self.assertEqual(1, len(result))
        self.assertIsInstance(result[0], sparse.csc_matrix)

    def test_fuse_representations(self):
        dv = DictVectorizer(sparse=False, sort=False)

        tfidf_result1 = {'word1': 1.546, 'word2': 1.467, 'word3': 0.55}
        doc_embedding_result1 = np.array([[0.98347, 1.384038, 7.1023803, 1.09854]])
        word_embedding_result1 = np.array([[0.123, 0.44561], [1.889, 3.22], [0.283, 0.887]])
        float_result1 = 8.8

        tfidf_result2 = {'word2': 1.467, 'word4': 1.1}
        doc_embedding_result2 = np.array([[2.331, 0.887, 1.1123, 0.7765]])
        word_embedding_result2 = np.array([[0.123, 0.44561], [5.554, 1.1234]])
        int_result2 = 7

        x = [[tfidf_result1, doc_embedding_result1, word_embedding_result1, float_result1],
             [tfidf_result2, doc_embedding_result2, word_embedding_result2, int_result2]]

        result = self.alg.fuse_representations(x, Centroid())

        dv.fit([tfidf_result1, tfidf_result2])
        centroid_word_embedding_1 = Centroid().combine(word_embedding_result1)
        centroid_word_embedding_2 = Centroid().combine(word_embedding_result2)

        expected_1 = np.hstack([dv.transform(tfidf_result1).flatten(), doc_embedding_result1.flatten(),
                                centroid_word_embedding_1.flatten(), float_result1])

        expected_2 = np.hstack([dv.transform(tfidf_result2).flatten(), doc_embedding_result2.flatten(),
                                centroid_word_embedding_2.flatten(), int_result2])

        self.assertTrue(all(isinstance(rep, np.ndarray) for rep in result))
        self.assertTrue(np.allclose(result[0], expected_1))
        self.assertTrue(np.allclose(result[1], expected_2))

    def test__load_available_contents(self):
        # test load_available_contents for content based algorithm
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

        interface_dict = self.alg._load_available_contents(movies_dir)
        self.assertIsInstance(interface_dict, LoadedContentsDict)

        interface_dict = self.alg._load_available_contents(movies_dir, {'tt0112281', 'tt0112302'})
        self.assertTrue(len(interface_dict) == 2)
        loaded_items_id_list = list(interface_dict)
        self.assertIn('tt0112281', loaded_items_id_list)
        self.assertTrue('tt0112302', loaded_items_id_list)

        # test load_available_contents for index
        index_alg = IndexQuery({'Plot': 'tfidf'})
        index_dir = os.path.join(dir_test_files, 'complex_contents', 'index')
        interface_dict = index_alg._load_available_contents(index_dir)
        self.assertIsInstance(interface_dict, LoadedContentsIndex)

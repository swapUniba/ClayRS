import os
from unittest import TestCase

from orange_cb_recsys.utils.load_content import load_content_instance

from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.centroid_vector import CentroidVector
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import CosineSimilarity
from orange_cb_recsys.utils.const import root_path

contents_path = os.path.join(root_path, 'contents')


class TestContentBasedAlgorithm(TestCase):

    def setUp(self) -> None:

        # ContentBasedAlgorithm is an abstract class, so we need to instantiate
        # a subclass to test its methods. No initialization since we are not testing
        # methods that need it
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
        movies_dir = os.path.join(contents_path, 'movies_codified/')

        content = load_content_instance(movies_dir, 'tt0112281')

        result = self.alg.extract_features_item(content)

        self.assertEqual(1, len(result))
        self.assertIsInstance(result[0], dict)

from random import random
from unittest import mock
from unittest.mock import patch, Mock, MagicMock
import numpy as np

from test.content_analyzer.embeddings.test_embedding_source import TestEmbeddingSource
from clayrs.content_analyzer.embeddings.embedding_loader.gensim import Gensim

result_matrix = {
    'title': np.array([random() for _ in range(25)]),
    'plot': np.array([random() for _ in range(25)])
}


def get_item(key):
    return result_matrix[key]


def similar_by_vector(vector, n_to_find):
    for i, vec in enumerate(result_matrix.values()):
        if np.array_equal(vec, vector):
            return [(list(result_matrix.keys())[i], 1)]


mocked_model = MagicMock()
mocked_model.__getitem__.side_effect = get_item
#mocked_model = MagicMock()
mocked_model.similar_by_vector.side_effect = similar_by_vector
mocked_model.vector_size = 25


class TestGensimDownloader(TestEmbeddingSource):

    def test_load(self):

        with mock.patch('gensim.downloader.info', return_value={'models': 'glove-twitter-25'}):
            with mock.patch('gensim.downloader.load', return_value=mocked_model):
                source = Gensim('glove-twitter-25')

        # result is a matrix containing 2 rows, one for 'title', one for 'plot'
        result = source.load(["title", "plot"])

        # the expected shape of result is (2, 25):
        # 2 for words and 25 due to the model 'glove-twitter-25'
        expected_shape = (2, 25)
        self.assertEqual(expected_shape, result.shape)

        self.assertWordEmbeddingMatches(source, result[0], "title")
        self.assertWordEmbeddingMatches(source, result[1], "plot")

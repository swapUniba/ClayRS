from random import random
from unittest import TestCase, mock
import numpy as np

from clayrs.content_analyzer.embeddings import Sbert

result_matrix = {
    'this is a phrase': np.array([random() for _ in range(768)]),
    'this is another phrase': np.array([random() for _ in range(768)])
}


def encode(sentence, show_progress_bar):
    return result_matrix[sentence]


class TestSbert(TestCase):

    @mock.patch('clayrs.content_analyzer.embeddings.sbert.SentenceTransformer')
    def test_sbert(self, mocked_model):
        instance = mocked_model.return_value
        instance.get_sentence_embedding_dimension.return_value = 768
        instance.encode.side_effect = encode

        source = Sbert()

        vector_size = source.get_vector_size()

        result = source.load(["this is a phrase", "this is another phrase"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(len(result[1]), vector_size)

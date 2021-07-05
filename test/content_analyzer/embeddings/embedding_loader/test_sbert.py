from unittest import TestCase
from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.sbert import Sbert


class TestSbert(TestCase):

    def test_sbert(self):
        source = Sbert()
        vector_size = source.get_vector_size()
        result = source.load(["this is a phrase", "this is another phrase"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)

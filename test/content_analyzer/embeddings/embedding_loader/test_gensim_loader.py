from test.content_analyzer.embeddings.test_embedding_source import TestEmbeddingSource
from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.gensim import Gensim


class TestGensimDownloader(TestEmbeddingSource):
    def test_load(self):
        source = Gensim('glove-twitter-25')

        # result is a matrix containing 2 rows, one for 'title', one for 'plot'
        result = source.load(["title", "plot"])

        # the expected shape of result is (2, 25):
        # 2 for words and 25 due to the model 'glove-twitter-25'
        expected_shape = (2, 25)
        self.assertEqual(expected_shape, result.shape)

        self.assertWordEmbeddingMatches(source, result[0], "title")
        self.assertWordEmbeddingMatches(source, result[1], "plot")

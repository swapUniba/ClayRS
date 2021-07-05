from orange_cb_recsys.content_analyzer.embeddings.embedding_loader.wiki2vec_loader import Wikipedia2VecLoader
from test.content_analyzer.embeddings.test_embedding_source import TestEmbeddingSource
import bz2


class TestWikipedia2VecDownloader(TestEmbeddingSource):
    def test_load(self):
        self.skipTest("SLOW")
        url = 'http://wikipedia2vec.s3.amazonaws.com/models/it/2018-04-20/itwiki_20180420_100d.pkl.bz2'
        #wget.download(url, '.')
        path = None
        with open('itwiki_20180420_100d.pkl.bz2', 'rb') as source, open('itwiki_20180420_100d.pkl', 'wb') as dest:
            path = dest.write(bz2.decompress(source.read()))

        source = Wikipedia2VecLoader('itwiki_20180420_100d.pkl')

        # result is a matrix containing 2 rows, one for 'notizie', one for 'nuove'
        result = source.load(['notizie', 'nuove'])

        # Now let's convert those 2 vector in words and
        # they should return 'title' and 'plot'.
        words = []
        for v in result:
            # 'most_similar_by_vector()' returns a list with top n
            # words similar to the vector given. I'm interested only in the most similar
            # so n = 1
            top_1 = source.model.most_similar_by_vector(v, 1)[0]
            words.append(top_1)

        # the expected shape of result is (2, 100):
        # 2 for words and 100 due to the model 'itwiki_20180420_100d'
        expected_shape = (2, 100)
        self.assertEqual(expected_shape, result.shape)

        # words[] contains [(<Word notizie>, 0.999...), (<Word nuove>, 1.000...)]
        # So I'm using indices to access the tuples values.
        # 'first_like' contains how similar is 'first_word' to the vector given result[0]
        first_word = (words[0])[0]
        first_like = (words[0])[1]
        second_word = (words[1])[0]
        second_like = (words[1])[1]

        # Obviously due to approximation the conversion won't return the
        # exact word, but if the likelihood it's equal to 1 with an approx of 'delta'
        # I'm assuming it's exactly that word
        # 'first_word' is an object so we must access the attribute 'text' to get the string
        self.assertEqual(first_word.text, "notizie")
        self.assertAlmostEqual(first_like, 1, delta=1e-6)
        self.assertEqual(second_word.text, "nuove")
        self.assertAlmostEqual(second_like, 1, delta=1e-6)

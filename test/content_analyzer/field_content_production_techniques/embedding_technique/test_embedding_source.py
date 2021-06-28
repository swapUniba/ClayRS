import bz2
import os
import numpy as np
from unittest import TestCase
from math import isclose

from orange_cb_recsys.content_analyzer.embedding_learner import GensimWord2Vec, GensimRandomIndexing, \
    GensimLatentSemanticAnalysis, GensimFastText, GensimDoc2Vec
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_source import \
    Gensim, Sbert, Wikipedia2VecLoader, EmbeddingSource
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, '../../../../datasets/movies_info_reduced.json')
doc2vec_file_path = os.path.join(THIS_DIR, "../../../../datasets/test_embedding_models/doc2vec_model.model")
lsa_file_path = os.path.join(THIS_DIR, "../../../../datasets/test_embedding_models/lsa/lsa_model.model")
ri_file_path = os.path.join(THIS_DIR, "../../../../datasets/test_embedding_models/ri_model.model")
word2vec_file_path = os.path.join(THIS_DIR, "../../../../datasets/test_embedding_models/word2vec_model.bin")


class WordEmbeddingAssertion:
    def assertWordEmbeddingMatches(self, source: EmbeddingSource, embedding: np.ndarray, word: str):
        # 'similar_by_vector()' returns a list with top n
        # words similar to the vector given. I'm interested only in the most similar
        # so n = 1
        # for example, top_1 will be in the following form ("title", 1.0)
        top_1 = source.model.similar_by_vector(embedding, 1)[0]

        # So I'm using indices to access the tuples values.
        # 'like' contains how similar is 'embedding_word' to the 'embedding' vector given
        embedding_word = top_1[0]
        like = top_1[1]

        # if the word associated with the embedding vector returned by the model doesn't match the word passed as
        # argument, AssertionError is raised
        if not embedding_word == word:
            raise AssertionError("Word %s is not %s" % (embedding_word, word))

        # Obviously due to approximation the conversion won't return the
        # exact word, but if the likelihood it's equal to 1 with a maximum error of 'abs_tol'
        # I'm assuming it's exactly that word
        if not isclose(like, 1, abs_tol=1e-6):
            raise AssertionError("Word %s and result word %s do not match" % (embedding_word, word))


class TestGensimDownloader(TestCase, WordEmbeddingAssertion):
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


class TestWikipedia2VecDownloader(TestCase):
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


class TestWordEmbeddingSourceGensimLearner(TestCase, WordEmbeddingAssertion):
    def test_doc2vec(self):
        source = GensimDoc2Vec(doc2vec_file_path)
        vector_size = 20
        result = source.load(["machine", "learning", "random_word"])

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)
        self.assertEqual(result[2].any(), False)

        self.assertWordEmbeddingMatches(source, result[0], "machine")
        self.assertWordEmbeddingMatches(source, result[1], "learning")

    def test_fasttext(self):
        # note: fasttext is trained because the resulting saved model file was too heavy
        source = GensimFastText("./test_source_fasttext", auto_save=False, min_count=1)
        source.fit(source=JSONFile(file_path), field_list=["Plot"], preprocessor_list=[NLTK()])
        vector_size = 100
        result = source.load(["first", "remote"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)

        self.assertWordEmbeddingMatches(source, result[0], "first")
        self.assertWordEmbeddingMatches(source, result[1], "remote")

    def test_lsa(self):
        source = GensimLatentSemanticAnalysis(lsa_file_path)
        vector_size = source.get_vector_size()
        result = source.load(["first", "remote", "random_word"])

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)
        self.assertEqual(result[2].any(), False)

    def test_ri(self):
        source = GensimRandomIndexing(ri_file_path)
        vector_size = 300
        result = source.load(["first", "remote", "random_word"])

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)
        self.assertEqual(result[2].any(), False)

    def test_word2vec(self):
        source = GensimWord2Vec(word2vec_file_path)
        vector_size = 100
        result = source.load(["first", "exile", "random_word"])

        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)
        self.assertEqual(result[2].any(), False)

        self.assertWordEmbeddingMatches(source, result[0], "first")
        self.assertWordEmbeddingMatches(source, result[1], "exile")


class TestSbert(TestCase):

    def test_sbert(self):
        source = Sbert()
        vector_size = source.get_vector_size()
        result = source.load(["this is a phrase", "this is another phrase"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), vector_size)
        self.assertEqual(result[1].any(), True)



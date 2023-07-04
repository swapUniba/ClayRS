from unittest import TestCase

import os

from clayrs.content_analyzer.information_processor.nltk_processor import NLTK
from clayrs.content_analyzer.raw_information_source import JSONFile
from test.content_analyzer.embeddings.test_embedding_source import TestEmbeddingSource
from clayrs.content_analyzer.embeddings.embedding_learner.doc2vec import GensimDoc2Vec
from clayrs.content_analyzer.embeddings.embedding_learner.fasttext import GensimFastText
from clayrs.content_analyzer.embeddings.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from clayrs.content_analyzer.embeddings.embedding_learner.random_indexing import GensimRandomIndexing
from clayrs.content_analyzer.embeddings.embedding_learner.word2vec import GensimWord2Vec
from test import dir_test_files


file_path = os.path.join(dir_test_files, 'movies_info_reduced.json')
doc2vec_file_path = os.path.join(dir_test_files, "test_embedding_models/doc2vec_model.kv")
lsa_file_path = os.path.join(dir_test_files, "test_embedding_models/lsa/lsa_model.model")
ri_file_path = os.path.join(dir_test_files, "test_embedding_models/ri_model.model")
word2vec_file_path = os.path.join(dir_test_files, "test_embedding_models/word2vec_model.kv")
fasttext_file_path = os.path.join(dir_test_files, "test_embedding_models/fasttext_model.kv")


class TestEmbeddingLearner(TestCase):
    def test_extract_corpus(self):
        preprocessor = NLTK(stopwords_removal=True, stemming=True)
        fields = ["Title", "Released"]
        expected = [['jumanji', '15', 'dec', '1995'],
                    ['grumpier', 'old', 'men', '22', 'dec', '1995'],
                    ['toy', 'stori', '22', 'nov', '1995'],
                    ['father', 'bride', 'part', 'ii', '08', 'dec', '1995'],
                    ['heat', '15', 'dec', '1995'],
                    ['tom', 'huck', '22', 'dec', '1995'],
                    ['wait', 'exhal', '22', 'dec', '1995'],
                    ['sabrina', '15', 'dec', '1995'],
                    ['dracula', ':', 'dead', 'love', '22', 'dec', '1995'],
                    ['nixon', '05', 'jan', '1996'],
                    ['american', 'presid', '17', 'nov', '1995'],
                    ['goldeney', '17', 'nov', '1995'],
                    ['balto', '22', 'dec', '1995'],
                    ['cutthroat', 'island', '22', 'dec', '1995'],
                    ['casino', '22', 'nov', '1995'],
                    ['sudden', 'death', '22', 'dec', '1995'],
                    ['sens', 'sensibl', '26', 'jan', '1996'],
                    ['four', 'room', '25', 'dec', '1995'],
                    ['money', 'train', '22', 'nov', '1995'],
                    ['ace', 'ventura', ':', 'natur', 'call', '10', 'nov', '1995']]

        src = JSONFile(file_path)
        learner = GensimLatentSemanticAnalysis("./test_extract_corpus")
        generated = learner.extract_corpus(src, fields, [preprocessor])

        self.assertEqual(generated, expected)


class TestWordEmbeddingSourceGensimLearner(TestEmbeddingSource):
    def test_doc2vec(self):
        # model created using d2c_test_data.json
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
        source = GensimFastText(fasttext_file_path)
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


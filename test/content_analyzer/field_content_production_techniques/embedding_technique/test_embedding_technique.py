from unittest import TestCase
import os

from clayrs.content_analyzer import BertTransformers
from clayrs.content_analyzer.content_representation.content import EmbeddingField
from clayrs.content_analyzer.embeddings.embedding_learner import GensimFastText
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique import \
    SentenceEmbeddingTechnique, Word2SentenceEmbedding, Word2DocEmbedding, \
    Sentence2DocEmbedding, WordEmbeddingTechnique, DocumentEmbeddingTechnique
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique \
    import Centroid
from clayrs.content_analyzer.embeddings.embedding_loader.gensim import Gensim
from clayrs.content_analyzer.embeddings.embedding_loader.sbert import Sbert
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.embedding_technique import \
    Sentence2WordEmbedding
from clayrs.content_analyzer.information_processor import NLTK
from clayrs.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, "movies_info_reduced.json")


class TestEmbeddingTechnique(TestCase):
    def test_produce_content(self):
        technique = WordEmbeddingTechnique(GensimFastText())
        embedding_list = technique.produce_content("Plot", [NLTK()], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

    def test_produce_content_str(self):
        self.skipTest("Test requires internet but is too complex to be mocked")
        technique = WordEmbeddingTechnique('glove-twitter-25')
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [NLTK()], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = SentenceEmbeddingTechnique('paraphrase-distilroberta-base-v1')
        self.assertIsInstance(technique.embedding_source, Sbert)
        embedding_list = technique.produce_content("Plot", [NLTK()], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = Word2DocEmbedding('glove-twitter-25', Centroid())
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = Sentence2DocEmbedding('paraphrase-distilroberta-base-v1', Centroid())
        self.assertIsInstance(technique.embedding_source, Sbert)
        embedding_list = technique.produce_content("Plot", [NLTK()], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = Word2SentenceEmbedding('glove-twitter-25', Centroid())
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

    def test_load_not_existing_source(self):
        self.skipTest("Test requires internet but is too complex to be mocked")
        with self.assertRaises(FileNotFoundError):
            WordEmbeddingTechnique('not_existing_model')

        with self.assertRaises(FileNotFoundError):
            WordEmbeddingTechnique('./not_existing_path')

        with self.assertRaises(FileNotFoundError):
            SentenceEmbeddingTechnique('not_existing_model')

        with self.assertRaises(FileNotFoundError):
            SentenceEmbeddingTechnique('./not_existing_path')

        with self.assertRaises(FileNotFoundError):
            DocumentEmbeddingTechnique('not_existing_model')

        with self.assertRaises(FileNotFoundError):
            DocumentEmbeddingTechnique('./not_existing_path')

        with self.assertRaises(FileNotFoundError):
            WordEmbeddingTechnique(Gensim('not_existing_model'))


class TestSentenceEmbeddingTechnique(TestCase):

    def test_produce_single_repr(self):
        file_path = os.path.join(dir_test_files, "movies_info_reduced.json")
        fromsentencetowords = SentenceEmbeddingTechnique(BertTransformers("prajjwal1/bert-tiny"))

        embedding_list = fromsentencetowords.produce_content("Plot", [], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)


class TestFromSentenceWordsEmbeddingTechnique(TestCase):

    def test_produce_single_repr(self):
        file_path = os.path.join(dir_test_files, "movies_info_reduced.json")
        fromsentencetowords = Sentence2WordEmbedding(BertTransformers("prajjwal1/bert-tiny"))

        embedding_list = fromsentencetowords.produce_content("Plot", [], [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

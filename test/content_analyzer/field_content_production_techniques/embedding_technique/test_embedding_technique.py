from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer.content_representation.content import EmbeddingField
from orange_cb_recsys.content_analyzer.embedding_learner import GensimFastText
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique import \
    SentenceEmbeddingTechnique, FromWordsSentenceEmbeddingTechnique, FromWordsDocumentEmbeddingTechnique, \
    FromSentencesDocumentEmbeddingTechnique, WordEmbeddingTechnique, DocumentEmbeddingTechnique
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique \
    import Centroid
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_source import \
    Gensim, Sbert
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../../datasets/movies_info_reduced.json")


class TestEmbeddingTechnique(TestCase):
    def test_produce_cotent(self):
        technique = WordEmbeddingTechnique(GensimFastText())
        embedding_list = technique.produce_content("Plot", [NLTK()], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

    def test_produce_content_str(self):
        technique = WordEmbeddingTechnique('glove-twitter-25')
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [NLTK()], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = SentenceEmbeddingTechnique('paraphrase-distilroberta-base-v1')
        self.assertIsInstance(technique.embedding_source, Sbert)
        embedding_list = technique.produce_content("Plot", [NLTK()], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = FromWordsDocumentEmbeddingTechnique('glove-twitter-25', Centroid())
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = FromSentencesDocumentEmbeddingTechnique('paraphrase-distilroberta-base-v1', Centroid())
        self.assertIsInstance(technique.embedding_source, Sbert)
        embedding_list = technique.produce_content("Plot", [NLTK()], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = FromWordsSentenceEmbeddingTechnique('glove-twitter-25', Centroid())
        self.assertIsInstance(technique.embedding_source, Gensim)
        embedding_list = technique.produce_content("Plot", [], JSONFile(file_path))
        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

    def test_load_not_existing_source(self):
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

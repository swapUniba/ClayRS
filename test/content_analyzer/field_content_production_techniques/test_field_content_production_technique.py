from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer.content_representation.content import EmbeddingField, StringField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_source import \
    GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EmbeddingTechnique, OriginalData, DefaultTechnique
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestEmbeddingTechnique(TestCase):
    def test_produce_content(self):
        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="doc")

        embedding_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="word")

        embedding_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)

        technique = EmbeddingTechnique(Centroid(), GensimDownloader('glove-twitter-25'), granularity="sentence")

        embedding_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(embedding_list), 20)
        self.assertIsInstance(embedding_list[0], EmbeddingField)


class TestOriginalData(TestCase):
    def test_produce_content(self):
        technique = OriginalData()

        data_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], StringField)


class TestDefaultTechnique(TestCase):
    def test_produce_content(self):
        technique = DefaultTechnique()

        data_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], StringField)

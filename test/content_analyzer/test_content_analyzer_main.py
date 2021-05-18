import os
from unittest import TestCase
import lzma
import pickle
import numpy as np

from orange_cb_recsys.content_analyzer.config import ExogenousConfig, ItemAnalyzerConfig
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique
from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig, FieldConfig
from orange_cb_recsys.content_analyzer.content_representation.content import StringField, FeaturesBagField, \
    EmbeddingField
from orange_cb_recsys.content_analyzer.field_content_production_techniques import EmbeddingTechnique, \
    Centroid, GensimDownloader
from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

filepath = '../../datasets/movies_info_reduced.json'
try:
    with open(filepath):
        pass
except FileNotFoundError:
    filepath = 'datasets/movies_info_reduced.json'


class TestContentsProducer(TestCase):
    def test_create_content(self):
        plot_config = FieldConfig(BabelPyEntityLinking())
        exogenous_config = ExogenousConfig(DBPediaMappingTechnique('Film', 'EN', 'Title'))
        content_analyzer_config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test")
        content_analyzer_config.add_single_config("Plot", plot_config)
        content_analyzer_config.add_single_exogenous(exogenous_config)
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()

    def test_field_exceptions(self):
        # test to make sure that the method that checks the field configs ids for each field name in the field_dict
        # of the content analyzer works. It considers the three cases this can occur: when passing the field_dict
        # with duplicate ids as argument for the content_analyzer, when setting the FieldConfig list with duplicates
        # for a specific field_name, and when appending a FieldConfig to the list associated with a specific field_name
        # but the config id is already in the list

        config_1 = FieldConfig(SkLearnTfIdf(), NLTK(), "test")
        config_2 = FieldConfig(SkLearnTfIdf(), NLTK(), "test")
        config_list = [config_1, config_2]
        field_dict = dict()
        field_dict["test"] = config_list

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test", False, field_dict)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test", False)
            config.add_multiple_config("test", config_list)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test", False)
            config.add_single_config("test", config_1)
            config.add_single_config("test", config_2)
            ContentAnalyzer(config).fit()

    def test_exogenous_exceptions(self):
        # test to make sure that the method that checks the exogenous configs ids in the exogenous_representation_list
        # of the content analyzer works. It considers the two cases this can occur: when passing the
        # exogenous_representation_list with duplicate ids as argument for the content_analyzer,
        # and when appending an ExogenousConfig to the list but the config id is already in the list

        config_1 = ExogenousConfig(DBPediaMappingTechnique('Film', 'EN', 'Title'), "test")
        config_2 = ExogenousConfig(DBPediaMappingTechnique('Film', 'EN', 'Title'), "test")
        exogenous_representation_list = [config_1, config_2]

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test", False,
                                        exogenous_representation_list=exogenous_representation_list)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(filepath), ["imdbID"], "movielens_test", False)
            config.add_single_exogenous(config_1)
            config.add_single_exogenous(config_2)
            ContentAnalyzer(config).fit()

    def test_create_content_tfidf(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(filepath),
            id='imdbID',
            output_directory="movielens_test_tfidf",
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig(SkLearnTfIdf())])

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_create_content_embedding(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(filepath),
            id=['imdbID'],
            output_directory="movielens_test_embedding",
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig(
                    EmbeddingTechnique(Centroid(), GensimDownloader(name='glove-twitter-25'), 'doc'),
                    NLTK(lemmatization=True, stopwords_removal=True))])

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_decode_field_data_string(self):
        filepath = '../../datasets/test_decode/movies_title_string.json'
        test_dir = '../../datasets/test_decode/'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/test_decode/movies_title_string.json'
            test_dir = 'datasets/test_decode/'

        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(filepath),
            id=['imdbID'],
            output_directory=test_dir + 'movies_string_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_string_' in str(name):

                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation(0), StringField)
                    self.assertIsInstance(content.get_field("Title").get_representation(0).value, str)
                    break

    def test_decode_field_data_tfidf(self):
        filepath = '../../datasets/test_decode/movies_title_tfidf.json'
        test_dir = '../../datasets/test_decode/'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/test_decode/movies_title_tfidf.json'
            test_dir = 'datasets/test_decode/'

        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(filepath),
            id=['imdbID'],
            output_directory=test_dir + 'movies_tfidf_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_tfidf_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation(0), FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title").get_representation(0).value, dict)
                    break

    def test_decode_field_data_embedding(self):
        filepath = '../../datasets/test_decode/movies_title_embedding.json'
        test_dir = '../../datasets/test_decode/'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/test_decode/movies_title_embedding.json'
            test_dir = 'datasets/test_decode/'

        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(filepath),
            id=['imdbID'],
            output_directory=test_dir + 'movies_embedding_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_embedding_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation(0), EmbeddingField)
                    self.assertIsInstance(content.get_field("Title").get_representation(0).value, np.ndarray)
                    break

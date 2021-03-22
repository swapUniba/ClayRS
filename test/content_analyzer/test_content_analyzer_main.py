import os
from unittest import TestCase
import lzma
import pickle
import numpy as np

from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig, FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.content_representation.content_field import StringField, FeaturesBagField, \
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
        entity_linking_pipeline = FieldRepresentationPipeline(BabelPyEntityLinking())
        plot_config = FieldConfig(None)
        plot_config.append_pipeline(entity_linking_pipeline)
        content_analyzer_config = ContentAnalyzerConfig('ITEM', JSONFile(filepath), ["imdbID"], "movielens_test")
        content_analyzer_config.append_field_config("Plot", plot_config)
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()

    def test_create_content_tfidf(self):
        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(filepath),
            id_field_name_list=['imdbID'],
            output_directory="movielens_test",
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[FieldRepresentationPipeline(
                    content_technique=SkLearnTfIdf())]
            ))

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

    def test_create_content_embedding(self):
        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(filepath),
            id_field_name_list=['imdbID'],
            output_directory="movielens_test",
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[

                    FieldRepresentationPipeline(
                            preprocessor_list=[NLTK(lemmatization=True, stopwords_removal=True)],
                            content_technique=EmbeddingTechnique(
                                                combining_technique=Centroid(),
                                                embedding_source=GensimDownloader(name='glove-twitter-25'),
                                                granularity='doc'))
                ]
            ))

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

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(filepath),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_string_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[
                    FieldRepresentationPipeline(content_technique=None),
                ]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_string_' in str(name):

                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), StringField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, str)
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

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(filepath),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_tfidf_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[
                    FieldRepresentationPipeline(content_technique=None),
                ]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_tfidf_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, dict)
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

        movies_ca_config = ContentAnalyzerConfig(
            content_type='Item',
            source=JSONFile(filepath),
            id_field_name_list=['imdbID'],
            output_directory=test_dir + 'movies_embedding_'
        )

        movies_ca_config.append_field_config(
            field_name='Title',
            field_config=FieldConfig(
                pipelines_list=[
                    FieldRepresentationPipeline(content_technique=None),
                ]

            )
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, name)) \
                    and 'movies_embedding_' in str(name):
                with lzma.open(os.path.join(test_dir, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title").get_representation('0'), EmbeddingField)
                    self.assertIsInstance(content.get_field("Title").get_representation('0').value, np.ndarray)
                    break

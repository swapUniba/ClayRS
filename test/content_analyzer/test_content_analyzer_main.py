import os
from unittest import TestCase
import lzma
import pickle
import numpy as np

from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique
from orange_cb_recsys.content_analyzer import ContentAnalyzer, FieldConfig, ExogenousConfig, ItemAnalyzerConfig
from orange_cb_recsys.content_analyzer.content_representation.content import SimpleField, FeaturesBagField, \
    EmbeddingField, IndexField
from orange_cb_recsys.content_analyzer.field_content_production_techniques import OriginalData
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique import Gensim
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.embedding_technique \
    import WordEmbeddingTechnique
from orange_cb_recsys.content_analyzer.field_content_production_techniques.entity_linking import BabelPyEntityLinking
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.memory_interfaces import SearchIndex, KeywordIndex
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(THIS_DIR, "../../datasets/movies_info_reduced.json")
decode_string = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_string.json")
decode_tfidf = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_tfidf.json")
decode_embedding = os.path.join(THIS_DIR, "../../datasets/test_decode/movies_title_embedding.json")
decode_path = os.path.join(THIS_DIR, '../../datasets/test_decode/')


class TestContentsProducer(TestCase):
    def test_create_content(self):
        plot_config = FieldConfig(BabelPyEntityLinking())
        exogenous_config = ExogenousConfig(DBPediaMappingTechnique('Film', 'EN', 'Title'))
        content_analyzer_config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test")
        content_analyzer_config.add_single_config("Title", plot_config)
        content_analyzer_config.add_single_exogenous(exogenous_config)
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()

        for name in os.listdir(THIS_DIR):
            if os.path.isdir(os.path.join(THIS_DIR, name)) \
                    and 'movielens_test' in str(name):

                with lzma.open(os.path.join(THIS_DIR, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title")[0].value, dict)
                    break

    def test_field_exceptions(self):
        # test to make sure that the method that checks the field configs ids for each field name in the field_dict
        # of the content analyzer works. It considers the three cases this can occur: when passing the field_dict
        # with duplicate ids as argument for the content_analyzer, when setting the FieldConfig list with duplicates
        # for a specific field_name, and when appending a FieldConfig to the list associated with a specific field_name
        # but the config id is already in the list

        config_1 = FieldConfig(SkLearnTfIdf(), NLTK(), id="test")
        config_2 = FieldConfig(SkLearnTfIdf(), NLTK(), id="test")
        config_list = [config_1, config_2]
        field_dict = dict()
        field_dict["test"] = config_list

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test", field_dict)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test")
            config.add_multiple_config("test", config_list)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test")
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
            config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test",
                                        exogenous_representation_list=exogenous_representation_list)
            ContentAnalyzer(config).fit()

        with self.assertRaises(ValueError):
            config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test")
            config.add_single_exogenous(config_1)
            config.add_single_exogenous(config_2)
            ContentAnalyzer(config).fit()

    def test_create_content_tfidf(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(movies_info_reduced),
            id='imdbID',
            output_directory="movielens_test_tfidf",
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig(SkLearnTfIdf())])

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

        for name in os.listdir(THIS_DIR):
            if os.path.isdir(os.path.join(THIS_DIR, name)) \
                    and 'movielens_test_tfidf' in str(name):

                with lzma.open(os.path.join(THIS_DIR, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title")[0].value, dict)
                    break

    def test_create_content_embedding(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(movies_info_reduced),
            id=['imdbID'],
            output_directory="movielens_test_embedding",
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig(
                    WordEmbeddingTechnique(Gensim('glove-twitter-25')),
                    NLTK(lemmatization=True, stopwords_removal=True))])

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

        for name in os.listdir(THIS_DIR):
            if os.path.isdir(os.path.join(THIS_DIR, name)) \
                    and 'movielens_test_embedding' in str(name):

                with lzma.open(os.path.join(THIS_DIR, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], EmbeddingField)
                    self.assertIsInstance(content.get_field("Title")[0].value, np.ndarray)
                    break

    def test_create_contents_in_index(self):
        output_dir = os.path.join(THIS_DIR, "movielens_test_original_index")
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(movies_info_reduced),
            id=['imdbID'],
            output_directory=output_dir,
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig(OriginalData(), NLTK(lemmatization=True, stopwords_removal=True),
                         SearchIndex(os.path.join(output_dir, "index")), "test_search"),

                         FieldConfig(SkLearnTfIdf(), NLTK(),
                                     KeywordIndex(os.path.join(output_dir, "index1")),
                                     "test_keyword"),

                         FieldConfig(OriginalData(), NLTK(),
                                     SearchIndex(os.path.join(output_dir, "index")))
                         ])

        content_analyzer = ContentAnalyzer(movies_ca_config)
        content_analyzer.fit()

        for name in os.listdir(THIS_DIR):
            if os.path.isdir(os.path.join(THIS_DIR, name)) \
                    and 'movielens_test_original_index' in str(name):

                with lzma.open(os.path.join(THIS_DIR, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], IndexField)
                    self.assertIsInstance(content.get_field("Title")[0].value, str)
                    self.assertIsInstance(content.get_field("Title")[1], IndexField)
                    self.assertIsInstance(content.get_field("Title")[1].value, str)
                    break

    def test_decode_field_data_string(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(decode_string),
            id=['imdbID'],
            output_directory=decode_path + 'movies_string_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(decode_path):
            if os.path.isdir(os.path.join(decode_path, name)) \
                    and 'movies_string_' in str(name):

                with lzma.open(os.path.join(decode_path, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], SimpleField)
                    self.assertIsInstance(content.get_field("Title")[0].value, str)
                    break

    def test_decode_field_data_tfidf(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(decode_tfidf),
            id=['imdbID'],
            output_directory=decode_path + 'movies_tfidf_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(decode_path):
            if os.path.isdir(os.path.join(decode_path, name)) \
                    and 'movies_tfidf_' in str(name):
                with lzma.open(os.path.join(decode_path, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], FeaturesBagField)
                    self.assertIsInstance(content.get_field("Title")[0].value, dict)
                    break

    def test_decode_field_data_embedding(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(decode_embedding),
            id=['imdbID'],
            output_directory=decode_path + 'movies_embedding_'
        )

        movies_ca_config.add_multiple_config(
            field_name='Title',
            config_list=[FieldConfig()]
        )
        ContentAnalyzer(config=movies_ca_config).fit()

        for name in os.listdir(decode_path):
            if os.path.isdir(os.path.join(decode_path, name)) \
                    and 'movies_embedding_' in str(name):
                with lzma.open(os.path.join(decode_path, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_field("Title")[0], EmbeddingField)
                    self.assertIsInstance(content.get_field("Title")[0].value, np.ndarray)
                    break

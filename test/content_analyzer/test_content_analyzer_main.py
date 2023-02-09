import os
import shutil
import unittest
from unittest import TestCase
import lzma
import pickle
import numpy as np
import scipy.sparse

from clayrs.content_analyzer.exogenous_properties_retrieval import PropertiesFromDataset
from clayrs.content_analyzer import ContentAnalyzer, FieldConfig, ExogenousConfig, ItemAnalyzerConfig
from clayrs.content_analyzer.content_representation.content import FeaturesBagField, \
    EmbeddingField, IndexField, PropertiesDict
from clayrs.content_analyzer.field_content_production_techniques import OriginalData
from clayrs.content_analyzer.embeddings.embedding_loader.gensim import Gensim
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.embedding_technique \
    import WordEmbeddingTechnique
from clayrs.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from clayrs.content_analyzer.information_processor import NLTK
from clayrs.content_analyzer.memory_interfaces import SearchIndex, KeywordIndex
from clayrs.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
movies_info_reduced = os.path.join(dir_test_files, "movies_info_reduced.json")

decode_path = os.path.join(dir_test_files, "test_decode/")
decode_string = os.path.join(decode_path, "movies_title_string.json")
decode_tfidf = os.path.join(decode_path, "movies_title_tfidf.json")
decode_embedding = os.path.join(decode_path, "movies_title_embedding.json")


class TestContentsProducer(TestCase):
    def test_create_content(self):
        exogenous_config = ExogenousConfig(PropertiesFromDataset(field_name_list=['Title']))
        content_analyzer_config = ItemAnalyzerConfig(JSONFile(movies_info_reduced), ["imdbID"], "movielens_test")
        content_analyzer_config.add_single_exogenous(exogenous_config)
        content_analyzer_config.add_single_exogenous(ExogenousConfig(PropertiesFromDataset(field_name_list=['Title'])))
        content_analyzer = ContentAnalyzer(content_analyzer_config)
        content_analyzer.fit()

        for name in os.listdir(THIS_DIR):
            if os.path.isdir(os.path.join(THIS_DIR, name)) \
                    and 'movielens_test' in str(name):

                with lzma.open(os.path.join(THIS_DIR, name, 'tt0113497.xz'), 'r') as file:
                    content = pickle.load(file)

                    self.assertIsInstance(content.get_exogenous_representation(0), PropertiesDict)
                    self.assertIsInstance(content.get_exogenous_representation(0).value, dict)
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

        config_1 = ExogenousConfig(PropertiesFromDataset(field_name_list=['Title']), "test")
        config_2 = ExogenousConfig(PropertiesFromDataset(field_name_list=['Title']), "test")
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
                    self.assertIsInstance(content.get_field("Title")[0].value, scipy.sparse.csc_matrix)
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
                    WordEmbeddingTechnique(Gensim('glove-wiki-gigaword-50')),
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
                         memory_interface=SearchIndex(os.path.join(output_dir, "index")), id="test_search"),

                         FieldConfig(SkLearnTfIdf(), NLTK(),
                                     memory_interface=KeywordIndex(os.path.join(output_dir, "index1")),
                                     id="test_keyword"),

                         FieldConfig(OriginalData(), NLTK(),
                                     memory_interface=SearchIndex(os.path.join(output_dir, "index")))
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


class TestContentAnalyzer(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.out_dir = 'test_export_json/'

    def test_fit_export_json(self):
        movies_ca_config = ItemAnalyzerConfig(
            source=JSONFile(movies_info_reduced),
            id=['imdbID'],
            output_directory=self.out_dir,
            export_json=True
        )

        movies_ca_config.add_single_config('Plot', FieldConfig(OriginalData()))
        movies_ca_config.add_single_config('Plot', FieldConfig(SkLearnTfIdf()))
        movies_ca_config.add_single_config('imdbRating', FieldConfig())

        ContentAnalyzer(movies_ca_config).fit()

        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'contents.json')))
        processed_source = list(JSONFile(os.path.join(self.out_dir, 'contents.json')))

        self.assertEqual(len(processed_source), 20)
        for processed_content in processed_source:
            self.assertIn('Plot#0', processed_content)
            self.assertIn('Plot#1', processed_content)
            self.assertIn('imdbRating#0', processed_content)

    def doCleanups(self) -> None:
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main()

from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer import ItemAnalyzerConfig
from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import WhooshTfIdf, SkLearnTfIdf
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestWhooshTfIdf(TestCase):
    def test_produce_content(self):
        technique = WhooshTfIdf()

        config = ItemAnalyzerConfig(
            source=JSONFile(file_path),
            id='imdbID',
            output_directory="test_whoosh_tf-idf",
        )

        features_bag_list = technique.produce_content("Plot", [], config)

        self.assertEqual(len(features_bag_list), 20)
        self.assertIsInstance(features_bag_list[0], FeaturesBagField)


class TestSkLearnTfIDF(TestCase):

    def test_produce_content(self):
        technique = SkLearnTfIdf()

        config = ItemAnalyzerConfig(
            source=JSONFile(file_path),
            id='imdbID',
            output_directory="test_SkLearn_tf-idf",
        )

        features_bag_list = technique.produce_content("Title", [], config)
        features_bag_list1 = technique.produce_content("Plot", [], config)

        self.assertEqual(len(features_bag_list), 20)
        self.assertIsInstance(features_bag_list[0], FeaturesBagField)

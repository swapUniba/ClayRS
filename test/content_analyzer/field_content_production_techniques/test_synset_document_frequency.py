from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.content_analyzer import ItemAnalyzerConfig
from orange_cb_recsys.content_analyzer.field_content_production_techniques import SynsetDocumentFrequency

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestSynsetDocumentFrequency(TestCase):
    def test_produce_content(self):
        technique = SynsetDocumentFrequency()

        config = ItemAnalyzerConfig(
            source=JSONFile(file_path),
            id='imdbID',
            output_directory="test_Synset",
        )

        features_bag_list = technique.produce_content("Title", [], config)

        self.assertEqual(len(features_bag_list), 20)
        self.assertIsInstance(features_bag_list[0], FeaturesBagField)

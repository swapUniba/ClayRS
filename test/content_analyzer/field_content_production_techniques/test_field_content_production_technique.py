from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer.content_representation.content import SimpleField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    OriginalData, DefaultTechnique
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestOriginalData(TestCase):
    def test_produce_content(self):
        technique = OriginalData()

        data_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], SimpleField)


class TestDefaultTechnique(TestCase):
    def test_produce_content(self):
        technique = DefaultTechnique()

        data_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], SimpleField)

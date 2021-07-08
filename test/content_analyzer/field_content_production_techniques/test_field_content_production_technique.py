import json
from unittest import TestCase
import os

from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.content_representation.content import SimpleField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    OriginalData, DefaultTechnique
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, DATFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/movies_info_reduced.json")


class TestOriginalData(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_name = 'test.dat'

    def test_produce_content(self):
        technique = OriginalData()

        data_list = technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], SimpleField)

    def test_produce_content_dtype_specified(self):
        technique = OriginalData(dtype=int)

        source = "50"
        with open(self.file_name, 'w') as f:
            f.write(source)

        result = technique.produce_content("0", [], DATFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, int)

    def test_produce_content_dtype_cant_convert(self):
        technique = OriginalData(dtype=int)

        source = "cant_convert"
        with open(self.file_name, 'w') as f:
            f.write(source)

        with self.assertRaises(ValueError):
            technique.produce_content("0", [], DATFile(self.file_name))

    def doCleanups(self) -> None:
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)


class TestDefaultTechnique(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.technique = DefaultTechnique()
        cls.file_name = 'test.json'

    def test_produce_content(self):

        data_list = self.technique.produce_content("Title", [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], SimpleField)

    def test_produce_content_int(self):
        source = [{"field": '50'}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, int)

        source = [{"field": 50}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, int)

    def test_produce_content_float(self):
        source = [{"field": '50.23'}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, float)

        source = [{"field": 50.23}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, float)

    def test_produce_content_list(self):
        source = [{"field": "['52ciao', '78999stringa']"}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, list)

        source = [{"field": ['52ciao', '78999stringa']}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        result = self.technique.produce_content("field", [], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, list)

    def test_produce_content_string(self):
        source = [{"field": "50"}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        # If preprocessor are specified, then we are sure the framework should import it as a str
        result = self.technique.produce_content("field", [NLTK(stopwords_removal=True)], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, str)

        source = [{"field": 50}]
        with open(self.file_name, 'w') as f:
            json.dump(source, f)

        # If preprocessor are specified, then we are sure the framework should import it as a str
        result = self.technique.produce_content("field", [NLTK(stopwords_removal=True)], JSONFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, str)

    def doCleanups(self) -> None:
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

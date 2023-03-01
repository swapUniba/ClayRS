import json
from unittest import TestCase
import os

import numpy as np

from clayrs.content_analyzer.content_representation.content import SimpleField
from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    OriginalData, FromNPY
from clayrs.content_analyzer.raw_information_source import JSONFile, DATFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, "movies_info_reduced.json")


class TestOriginalData(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.file_name = 'test.dat'

    def test_produce_content(self):
        technique = OriginalData()

        data_list = technique.produce_content("Title", [], [], JSONFile(file_path))

        self.assertEqual(len(data_list), 20)
        self.assertIsInstance(data_list[0], SimpleField)

    def test_produce_content_dtype_specified(self):
        technique = OriginalData(dtype=int)

        source = "50"
        with open(self.file_name, 'w') as f:
            f.write(source)

        result = technique.produce_content("0", [], [], DATFile(self.file_name))

        self.assertIsInstance(result[0], SimpleField)
        self.assertIsInstance(result[0].value, int)

    def test_produce_content_dtype_cant_convert(self):
        technique = OriginalData(dtype=int)

        source = "cant_convert"
        with open(self.file_name, 'w') as f:
            f.write(source)

        with self.assertRaises(ValueError):
            technique.produce_content("0", [], [], DATFile(self.file_name))

    def doCleanups(self) -> None:
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)


class TestFromNPY(TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # setup 3 arrays with different dimensionalities

        array_0_dim = np.array([])

        array_1_dim = np.array([
            0, 0, 1, 1, 2, 2, 2, 2, 2
        ])

        array_2_dim = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])

        path_array_0_dim = 'test_array_0_dim.npy'
        path_array_1_dim = 'test_array_1_dim.npy'
        path_array_2_dim = 'test_array_2_dim.npy'

        np.save(path_array_0_dim, array_0_dim)
        np.save(path_array_1_dim, array_1_dim)
        np.save(path_array_2_dim, array_2_dim)

        cls.path_array_0_dim = path_array_0_dim
        cls.path_array_1_dim = path_array_1_dim
        cls.path_array_2_dim = path_array_2_dim

        # source containing an idx field matching the row from the numpy feature matrix associated to the item

        idxs_source_path = 'tmp_source_array.json'

        data = [
            {'id': 0, 'idx': 0},
            {'id': 1, 'idx': 1},
            {'id': 2, 'idx': 2}
        ]

        with open(idxs_source_path, 'w') as f:
            json.dump(data, f)

        cls.idxs_source_path = idxs_source_path

    def test_produce_content(self):

        idxs_source = JSONFile(self.idxs_source_path)

        # since the array is empty it would be impossible to extract elements from it, ValueError should be raised
        with self.assertRaises(ValueError):
            FromNPY(self.path_array_0_dim)

        # since the array is 1 dimensional, outputs should be single values
        technique = FromNPY(self.path_array_1_dim)
        output_1_dim = technique.produce_content('idx', [], [], idxs_source)

        self.assertEqual(3, len(output_1_dim))
        self.assertEqual(0, technique._missing)
        self.assertEqual(0, output_1_dim[0].value)
        self.assertEqual(0, output_1_dim[1].value)
        self.assertEqual(1, output_1_dim[2].value)

        # since the array is 2 dimensional, outputs should be arrays
        technique = FromNPY(self.path_array_2_dim)
        output_2_dim = technique.produce_content('idx', [], [], idxs_source)

        self.assertEqual(3, len(output_2_dim))
        self.assertEqual(0, technique._missing)
        np.testing.assert_array_equal(np.array([0, 0]), output_2_dim[0].value)
        np.testing.assert_array_equal(np.array([1, 1]), output_2_dim[1].value)
        np.testing.assert_array_equal(np.array([2, 2]), output_2_dim[2].value)

    def test_produce_content_idxs_out_of_bounds(self):

        # test limit case in which an index greater than the number of rows in the numpy feature matrix is provided

        data = [
            {'id': 0, 'idx': 0},
            {'id': 1, 'idx': 1},
            {'id': 2, 'idx': 999},
        ]

        idxs_out_of_bounds_source_path = 'tmp_source_array_out_of_bounds.npy'

        with open(idxs_out_of_bounds_source_path, 'w') as f:
            json.dump(data, f)

        source_out_of_bounds = JSONFile(idxs_out_of_bounds_source_path)
        technique = FromNPY(self.path_array_2_dim)
        with self.assertRaises(IndexError):
            technique.produce_content('idx', [], [], source_out_of_bounds)

        os.remove(idxs_out_of_bounds_source_path)

    def test_produce_content_idxs_not_int(self):

        # test limit case in which a wrong type of index is provided (str that cannot be converted to int in this case)

        data = [
            {'id': 0, 'idx': 0},
            {'id': 1, 'idx': 1},
            {'id': 2, 'idx': 'test'},
        ]

        idxs_not_int_path = 'tmp_source_array_out_of_bounds.npy'

        with open(idxs_not_int_path, 'w') as f:
            json.dump(data, f)

        source_out_of_bounds = JSONFile(idxs_not_int_path)
        technique = FromNPY(self.path_array_2_dim)

        with self.assertRaises(ValueError):
            technique.produce_content('idx', [], [], source_out_of_bounds)

        os.remove(idxs_not_int_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.path_array_0_dim)
        os.remove(cls.path_array_1_dim)
        os.remove(cls.path_array_2_dim)
        os.remove(cls.idxs_source_path)

# DECODE POSSIBLE REPRESENTATION: Not implemented for now
#
# class TestDefaultTechnique(TestCase):
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.technique = DefaultTechnique()
#         cls.file_name = 'test.json'
#
#     def test_produce_content(self):
#
#         data_list = self.technique.produce_content("Title", [], JSONFile(file_path))
#
#         self.assertEqual(len(data_list), 20)
#         self.assertIsInstance(data_list[0], SimpleField)
#
#     def test_produce_content_int(self):
#         source = [{"field": '50'}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, int)
#
#         source = [{"field": 50}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, int)
#
#     def test_produce_content_float(self):
#         source = [{"field": '50.23'}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, float)
#
#         source = [{"field": 50.23}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, float)
#
#     def test_produce_content_list(self):
#         source = [{"field": "['52ciao', '78999stringa']"}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, list)
#
#         source = [{"field": ['52ciao', '78999stringa']}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         result = self.technique.produce_content("field", [], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, list)
#
#     def test_produce_content_string(self):
#         source = [{"field": "50"}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         # If preprocessor are specified, then we are sure the framework should import it as a str
#         result = self.technique.produce_content("field", [NLTK(stopwords_removal=True)], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, str)
#
#         source = [{"field": 50}]
#         with open(self.file_name, 'w') as f:
#             json.dump(source, f)
#
#         # If preprocessor are specified, then we are sure the framework should import it as a str
#         result = self.technique.produce_content("field", [NLTK(stopwords_removal=True)], JSONFile(self.file_name))
#
#         self.assertIsInstance(result[0], SimpleField)
#         self.assertIsInstance(result[0].value, str)
#
#     def doCleanups(self) -> None:
#         if os.path.isfile(self.file_name):
#             os.remove(self.file_name)

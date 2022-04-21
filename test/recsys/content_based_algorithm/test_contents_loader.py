import os
import unittest
from os import listdir
from os.path import splitext, isfile, join

from clayrs.content_analyzer import SearchIndex
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict, LoadedContentsIndex
from test import dir_test_files


class TestLoadedContentsDict(unittest.TestCase):
    def test_all(self):
        # test load_available_contents for content based algorithm
        movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')

        interface_dict = LoadedContentsDict(movies_dir)

        # we are testing get_contents_interface
        self.assertIsInstance(interface_dict.get_contents_interface(), dict)

        # since we didn't specify which items to load, we expected it has loaded all items from the folder
        expected = {splitext(filename)[0]
                    for filename in listdir(movies_dir)
                    if isfile(join(movies_dir, filename)) and splitext(filename)[1] == ".xz"}

        # we are testing also iter
        result = set(interface_dict)
        self.assertEqual(expected, result)

        # test loaded contents specified
        interface_dict = LoadedContentsDict(movies_dir, {'tt0112281', 'tt0112302'})

        # we are testing len
        self.assertTrue(len(interface_dict) == 2)

        # we are testing getitem
        self.assertIsNotNone(interface_dict['tt0112281'])
        self.assertIsNotNone(interface_dict['tt0112302'])

        # we are testing get
        self.assertIsNotNone(interface_dict.get('tt0112281'))
        self.assertIsNone(interface_dict.get('should be None'))


class TestLoadedContentsIndex(unittest.TestCase):
    def test_all(self):
        index = "../test/test_files/index"

        self.assertIsInstance(LoadedContentsIndex(index).get_contents_interface(), SearchIndex)

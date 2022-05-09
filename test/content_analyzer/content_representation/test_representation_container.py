from unittest import TestCase
import numpy as np

from clayrs.content_analyzer.content_representation.representation_container import RepresentationContainer


class TestRepresentationContainer(TestCase):

    def test_rep_container(self):
        rep_container = RepresentationContainer(['rep1', 'rep2', 'rep3'], ['test1', None, 'test3'])

        # tests to check that the indexes and columns of the dataframe in representation container are set as expected
        self.assertEqual([0, 1, 2], rep_container.get_internal_index())
        self.assertEqual(['test1', None, 'test3'], rep_container.get_external_index())
        self.assertEqual(['rep1', 'rep2', 'rep3'], rep_container.get_representations())

        # tests to check that the representation related to the internal_id or external_id passed to rep_container
        # is the appropriate representation
        for _ in range(15000):
            self.assertEqual('rep1', rep_container[0])
            self.assertEqual('rep3', rep_container['test3'])

        # tests to check the correct functionality of the append and remove method
        rep_container.append('rep4', 'test4')
        self.assertEqual('rep4', rep_container['test4'])

        value_removed = rep_container.pop('test4')
        self.assertEqual('rep4', value_removed)
        self.assertFalse('rep4' in rep_container.get_representations())

        # test for empty representation container
        empty_rep_container = RepresentationContainer()
        self.assertEqual(0, len(empty_rep_container))

        # test for passing single value to representation container constructor instead of lists
        single_rep_container = RepresentationContainer('rep', 'test')
        self.assertEqual('rep', single_rep_container['test'])

        # test exception different length of external_id representation lists when passed to the constructor
        with self.assertRaises(ValueError):
            RepresentationContainer(['rep1', 'rep2'], ['test1'])

        # test exception different length of external_id representation lists when passed to the 'append' method
        with self.assertRaises(ValueError):
            rep_container.append(['rep1', 'rep2'], ['test1'])

        # test exception representation not present
        with self.assertRaises(KeyError):
            err = rep_container['not_existent']

    def test_iter(self):
        rep_container = RepresentationContainer(['rep1', 'rep2', 'rep3'], ['test1', None, 'test3'])

        expected_list = [
            {'internal_id': 0, 'external_id': 'test1', 'representation': 'rep1'},
            {'internal_id': 1, 'external_id': None, 'representation': 'rep2'},
            {'internal_id': 2, 'external_id': 'test3', 'representation': 'rep3'}
        ]

        it = iter(rep_container)

        self.assertEqual(expected_list[0], next(it))
        self.assertEqual(expected_list[1], next(it))
        self.assertEqual(expected_list[2], next(it))

        # Check that the iterator gives an error since there aren't any items left
        with self.assertRaises(StopIteration):
            next(it)

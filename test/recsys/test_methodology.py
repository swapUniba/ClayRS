import unittest
from unittest import TestCase
import pandas as pd
import numpy as np

from clayrs.content_analyzer import Ratings
from clayrs.recsys.methodology import TestRatingsMethodology, TestItemsMethodology, \
    TrainingItemsMethodology, AllItemsMethodology

train1 = pd.DataFrame.from_records([
    ("001", "tt0112281", 3.5, "54654675"),
    ("001", "tt0112302", 4.5, "54654675"),
    ("002", "tt0112346", 4, "54654675"),
    ("003", "tt0112453", 2, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

test1 = pd.DataFrame.from_records([
    ("001", "tt0112641", 2, "54654675"),
    ("001", "tt0112760", 1, "54654675"),
    ("002", "tt0112641", 3, "54654675"),
    ("002", "tt0112896", 2, "54654675"),
    ("003", "tt0113041", 3, "54654675"),
    ("003", "tt0112281", 5, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

train2 = pd.DataFrame.from_records([
    ("001", "tt0112641", 2, "54654675"),
    ("001", "tt0112760", 1, "54654675"),
    ("002", "tt0112641", 3, "54654675"),
    ("002", "tt0112896", 2, "54654675"),
    ("003", "tt0113041", 3, "54654675"),
    ("003", "tt0112281", 5, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

test2 = pd.DataFrame.from_records([
    ("001", "tt0112281", 3.5, "54654675"),
    ("001", "tt0112302", 4.5, "54654675"),
    ("002", "tt0112346", 4, "54654675"),
    ("003", "tt0112453", 2, "54654675")],
    columns=["from_id", "to_id", "score", "timestamp"])

train1_rat = Ratings.from_dataframe(train1)
test1_rat = Ratings.from_dataframe(test1)
train2_rat = Ratings.from_dataframe(train2)
test2_rat = Ratings.from_dataframe(test2)


def _unpack_result_w_iter(result_list_w_iter):
    result_list = []
    for result_w_iter in result_list_w_iter:
        result_unpacked = {}
        for user, items_iterator in result_w_iter.items():
            result_unpacked[user] = set(items_iterator)

        result_list.append(result_unpacked)

    return result_list


class TestTestRatingsMethodology(TestCase):

    def test_filter_all(self):

        result_list = [TestRatingsMethodology().filter_all(train1_rat, test1_rat),
                       TestRatingsMethodology().filter_all(train2_rat, test2_rat)]

        # for every user get the items in its test_set1
        expected_list = [test1[['from_id', 'to_id']], test2[['from_id', 'to_id']]]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_filter_all_only_greater_eq(self):

        result_list = [TestRatingsMethodology(only_greater_eq=3).filter_all(train1_rat, test1_rat),
                       TestRatingsMethodology(only_greater_eq=3).filter_all(train2_rat, test2_rat)]

        # for every user get the items in its test_set1 with score >= 3
        expected_split_1 = pd.DataFrame({
            'from_id': ['002',
                        '003', '003'],
            'to_id': ["tt0112641",
                      "tt0113041", "tt0112281"]
        })

        # for every user get the items in its test_set2 with score >= 3
        expected_split_2 = pd.DataFrame({
            'from_id': ['001', '001',
                        '002'],
            'to_id': ["tt0112281", "tt0112302",
                      "tt0112346"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_result_as_dict_iter(self):
        result_list_iter = [TestRatingsMethodology().filter_all(train1_rat, test1_rat, result_as_iter_dict=True),
                            TestRatingsMethodology().filter_all(train2_rat, test2_rat, result_as_iter_dict=True)]

        # for every user get the items in its test_set1
        expected_list = [{'001': {"tt0112641", "tt0112760"},
                          '002': {"tt0112641", "tt0112896"},
                          '003': {"tt0113041", "tt0112281"}},
                         {'001': {"tt0112281", "tt0112302"},
                          '002': {"tt0112346"},
                          '003': {"tt0112453"}}
                         ]

        result_list = _unpack_result_w_iter(result_list_iter)

        self.assertTrue(len(expected_list), len(result_list))

        self.assertCountEqual(result_list, expected_list)


# poor choice of words sadly
class TestTestItemsMethodology(TestCase):

    def test_filter_all(self):

        result_list = [TestItemsMethodology().filter_all(train1_rat, test1_rat),
                       TestItemsMethodology().filter_all(train2_rat, test2_rat)]

        # for every user get the all items present in test_set1 except the items
        # present in the training_set1 of the user
        expected_split_1 = pd.DataFrame({
            'from_id': ['001', '001', '001', '001',
                        '002', '002', '002', '002', '002',
                        '003', '003', '003', '003', '003'],
            'to_id': ["tt0112641", "tt0112760", "tt0112896", "tt0113041",
                      "tt0112641", "tt0112760", "tt0112896", "tt0113041", "tt0112281",
                      "tt0112641", "tt0112760", "tt0112896", "tt0113041", "tt0112281"]
        })

        # for every user get the all items present in test_set2 except the items
        # present in the training_set2 of the user
        expected_split_2 = pd.DataFrame({
            'from_id': ['001', '001', '001', '001',
                        '002', '002', '002', '002',
                        '003', '003', '003'],
            'to_id': ["tt0112281", "tt0112302", "tt0112346", "tt0112453",
                      "tt0112281", "tt0112302", "tt0112346", "tt0112453",
                      "tt0112302", "tt0112346", "tt0112453"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_filter_all_only_greater_eq(self):
        result_list = [TestItemsMethodology(only_greater_eq=3).filter_all(train1_rat, test1_rat),
                       TestItemsMethodology(only_greater_eq=3).filter_all(train2_rat, test2_rat)]

        # for every user get the all items present in test_set1 with score >= 3 except the items
        # present in the training_set1 of the user
        expected_split_1 = pd.DataFrame({
            'from_id': ['001', '001',
                        '002', '002', '002',
                        '003', '003', '003'],
            'to_id': ["tt0112641", "tt0113041",
                      "tt0112641", "tt0113041", "tt0112281",
                      "tt0112641", "tt0113041", "tt0112281"]
        })

        # for every user get the all items present in test_set2 with score >= 3 except the items
        # present in the training_set2 of the user
        expected_split_2 = pd.DataFrame({
            'from_id': ['001', '001', '001',
                        '002', '002', '002',
                        '003', '003'],
            'to_id': ["tt0112281", "tt0112302", "tt0112346",
                      "tt0112281", "tt0112302", "tt0112346",
                      "tt0112302", "tt0112346"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_result_as_dict(self):
        result_list_iter = [TestItemsMethodology().filter_all(train1_rat, test1_rat, result_as_iter_dict=True),
                            TestItemsMethodology().filter_all(train2_rat, test2_rat, result_as_iter_dict=True)]

        expected_split_1 = {
            '001': ["tt0112641", "tt0112760", "tt0112896", "tt0113041"],
            '002': ["tt0112641", "tt0112760", "tt0112896", "tt0113041", "tt0112281"],
            '003': ["tt0112641", "tt0112760", "tt0112896", "tt0113041", "tt0112281"]
        }

        expected_split_2 = {
            '001': ["tt0112281", "tt0112302", "tt0112346", "tt0112453"],
            '002': ["tt0112281", "tt0112302", "tt0112346", "tt0112453"],
            '003': ["tt0112302", "tt0112346", "tt0112453"]
        }

        expected_list = [expected_split_1, expected_split_2]
        result_list = _unpack_result_w_iter(result_list_iter)

        self.assertTrue(len(expected_list), len(result_list))

        self.assertCountEqual(expected_list[0], result_list[0])
        self.assertCountEqual(expected_list[1], result_list[1])


class TestTrainingItemsMethodology(TestCase):

    def test_filter_all(self):

        result_list = [TrainingItemsMethodology().filter_all(train1_rat, test1_rat),
                       TrainingItemsMethodology().filter_all(train2_rat, test2_rat)]

        # for every user get the all items present in training_set1 except the items
        # present in the training_set1 of the user
        expected_split_1 = pd.DataFrame({
            'from_id': ['001', '001',
                        '002', '002', '002',
                        '003', '003', '003'],
            'to_id': ["tt0112346", "tt0112453",
                      "tt0112281", "tt0112302", "tt0112453",
                      "tt0112281", "tt0112302", "tt0112346"]
        })

        # for every user get the all items present in training_set2 except the items
        # present in the training_set2 of the user
        expected_split_2 = pd.DataFrame({
            'from_id': ['001', '001', '001',
                        '002', '002', '002',
                        '003', '003', '003'],
            'to_id': ["tt0112896", "tt0113041", "tt0112281",
                      "tt0112760", "tt0113041", "tt0112281",
                      "tt0112641", "tt0112760", "tt0112896"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_filter_all_only_greater_eq(self):
        result_list = [TrainingItemsMethodology(only_greater_eq=3).filter_all(train1_rat, test1_rat),
                       TrainingItemsMethodology(only_greater_eq=3).filter_all(train2_rat, test2_rat)]

        # for every user get the all items present in training_set1 with score >= 3 except the items
        # present in the training_set1 of the user
        expected_split_1 = pd.DataFrame({
            'from_id': ['001',
                        '002', '002',
                        '003', '003', '003'],
            'to_id': ["tt0112346",
                      "tt0112281", "tt0112302",
                      "tt0112281", "tt0112302", "tt0112346"]
        })

        # for every user get the all items present in training_set2 with score >= 3 except the items
        # present in the training_set2 of the user
        expected_split_2 = pd.DataFrame({
            'from_id': ['001', '001',
                        '002', '002',
                        '003'],
            'to_id': ["tt0113041", "tt0112281",
                      "tt0113041", "tt0112281",
                      "tt0112641"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_result_as_dict(self):
        result_list_iter = [TrainingItemsMethodology().filter_all(train1_rat, test1_rat, result_as_iter_dict=True),
                            TrainingItemsMethodology().filter_all(train2_rat, test2_rat, result_as_iter_dict=True)]

        expected_split_1 = {
            '001': ["tt0112346", "tt0112453"],
            '002': ["tt0112281", "tt0112302", "tt0112453"],
            '003': ["tt0112281", "tt0112302", "tt0112346"]
        }

        expected_split_2 = {
            '001': ["tt0112896", "tt0113041", "tt0112281"],
            '002': ["tt0112760", "tt0113041", "tt0112281"],
            '003': ["tt0112641", "tt0112760", "tt0112896"]
        }

        expected_list = [expected_split_1, expected_split_2]
        result_list = _unpack_result_w_iter(result_list_iter)

        self.assertTrue(len(expected_list), len(result_list))

        self.assertCountEqual(expected_list[0], result_list[0])
        self.assertCountEqual(expected_list[1], result_list[1])


class TestAllItemsMethodology(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.all_items = ["tt0112281",
                         "tt0112302",
                         "tt0112346",
                         "tt0112453",
                         "iall1",
                         "iall2",
                         "iall3",
                         "iall4"]

    def test_filter_all(self):
        result_list = [AllItemsMethodology(set(self.all_items)).filter_all(train1_rat, test1_rat),
                       AllItemsMethodology(set(self.all_items)).filter_all(train2_rat, test2_rat)]

        expected_split_1 = pd.DataFrame({
            'from_id': ["001", "001", "001", "001", "001", "001",
                        "002", "002", "002", "002", "002", "002", "002",
                        "003", "003", "003", "003", "003", "003", "003"],
            'to_id': ["tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4",
                      "tt0112281", "tt0112302", "tt0112453", "iall1", "iall2", "iall3", "iall4",
                      "tt0112281", "tt0112302", "tt0112346", "iall1", "iall2", "iall3", "iall4", ]
        })

        expected_split_2 = pd.DataFrame({
            'from_id': ["001", "001", "001", "001", "001", "001", "001", "001",
                        "002", "002", "002", "002", "002", "002", "002", "002",
                        "003", "003", "003", "003", "003", "003", "003"],
            'to_id': ["tt0112281", "tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4",
                      "tt0112281", "tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4",
                      "tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4"]
        })

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):
            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_result_as_dict(self):
        result_list_iter = [AllItemsMethodology(set(self.all_items)).filter_all(train1_rat, test1_rat,
                                                                                result_as_iter_dict=True),
                            AllItemsMethodology(set(self.all_items)).filter_all(train2_rat, test2_rat,
                                                                                result_as_iter_dict=True)]

        expected_split_1 = {
            '001': ["tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4"],
            '002': ["tt0112281", "tt0112302", "tt0112453", "iall1", "iall2", "iall3", "iall4"],
            '003': ["tt0112281", "tt0112302", "tt0112346", "iall1", "iall2", "iall3", "iall4"]
        }

        expected_split_2 = {
            '001': ["tt0112281", "tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4", ],
            '002': ["tt0112281", "tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4", ],
            '003': ["tt0112302", "tt0112346", "tt0112453", "iall1", "iall2", "iall3", "iall4"]
        }

        expected_list = [expected_split_1, expected_split_2]
        result_list = _unpack_result_w_iter(result_list_iter)

        self.assertTrue(len(expected_list), len(result_list))

        self.assertCountEqual(expected_list[0], result_list[0])
        self.assertCountEqual(expected_list[1], result_list[1])


if __name__ == "__main__":
    unittest.main()

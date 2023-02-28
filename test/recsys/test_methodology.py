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

# we create manually the mapping since we want a global mapping containing train and test items
item1_map = {}
for item_id in train1[["to_id"]].append(test1[["to_id"]])["to_id"]:
    if item_id not in item1_map:
        item1_map[item_id] = len(item1_map)

item2_map = {}
for item_id in train2[["to_id"]].append(test2[["to_id"]])["to_id"]:
    if item_id not in item2_map:
        item2_map[item_id] = len(item2_map)

train1_rat = Ratings.from_dataframe(train1, item_map=item1_map)
test1_rat = Ratings.from_dataframe(test1, item_map=item1_map)
train2_rat = Ratings.from_dataframe(train2, item_map=item2_map)
test2_rat = Ratings.from_dataframe(test2, item_map=item2_map)


class TestMethodology(TestCase):

    def assertDictListCountEqual(self, dict1, dict2):

        user_list_dict1 = list(dict1.keys())
        user_list_dict2 = list(dict2.keys())
        self.assertCountEqual(user_list_dict1, user_list_dict2)

        for user in user_list_dict1:
            self.assertCountEqual(dict1[user], dict2[user])


class TestTestRatingsMethodology(TestMethodology):

    def test_filter_all(self):

        ratings_1 = TestRatingsMethodology()
        ratings_2 = TestRatingsMethodology()

        ratings_1.setup(train1_rat, test1_rat)
        ratings_2.setup(train2_rat, test2_rat)

        result_list = [ratings_1.filter_all(train1_rat, test1_rat, ids_as_str=True),
                       ratings_2.filter_all(train2_rat, test2_rat, ids_as_str=True)]

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

        ratings_1 = TestRatingsMethodology(only_greater_eq=3)
        ratings_2 = TestRatingsMethodology(only_greater_eq=3)

        ratings_1.setup(train1_rat, test1_rat)
        ratings_2.setup(train2_rat, test2_rat)

        result_list = [ratings_1.filter_all(train1_rat, test1_rat, ids_as_str=True),
                       ratings_2.filter_all(train2_rat, test2_rat, ids_as_str=True)]

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

        ratings_1 = TestRatingsMethodology()
        ratings_2 = TestRatingsMethodology()

        ratings_1.setup(train1_rat, test1_rat)
        ratings_2.setup(train2_rat, test2_rat)

        result_list = [ratings_1.filter_all(train1_rat, test1_rat, result_as_dict=True, ids_as_str=True),
                       ratings_2.filter_all(train2_rat, test2_rat, result_as_dict=True, ids_as_str=True)]

        # convert numpy to list for dict equal assertion
        result_list[0] = dict((user, list(filter_list)) for user, filter_list in result_list[0].items())
        result_list[1] = dict((user, list(filter_list)) for user, filter_list in result_list[1].items())

        # for every user get the items in its test_set1
        expected_list = [{'001': ["tt0112641", "tt0112760"],
                          '002': ["tt0112641", "tt0112896"],
                          '003': ["tt0113041", "tt0112281"]},

                         {'001': ["tt0112281", "tt0112302"],
                          '002': ["tt0112346"],
                          '003': ["tt0112453"]}
                         ]

        self.assertTrue(len(expected_list), len(result_list))

        self.assertDictListCountEqual(expected_list[0], result_list[0])
        self.assertDictListCountEqual(expected_list[1], result_list[1])


# poor choice of words sadly
class TestTestItemsMethodology(TestMethodology):

    def test_filter_all(self):

        test_1 = TestItemsMethodology()
        test_2 = TestItemsMethodology()

        test_1.setup(train1_rat, test1_rat)
        test_2.setup(train2_rat, test2_rat)

        result_list = [test_1.filter_all(train1_rat, test1_rat, ids_as_str=True),
                       test_2.filter_all(train2_rat, test2_rat, ids_as_str=True)]

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

        test_1 = TestItemsMethodology(only_greater_eq=3)
        test_2 = TestItemsMethodology(only_greater_eq=3)

        test_1.setup(train1_rat, test1_rat)
        test_2.setup(train2_rat, test2_rat)

        result_list = [test_1.filter_all(train1_rat, test1_rat, ids_as_str=True),
                       test_2.filter_all(train2_rat, test2_rat, ids_as_str=True)]

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

        test_1 = TestItemsMethodology()
        test_2 = TestItemsMethodology()

        test_1.setup(train1_rat, test1_rat)
        test_2.setup(train2_rat, test2_rat)

        result_list = [test_1.filter_all(train1_rat, test1_rat, result_as_dict=True),
                       test_2.filter_all(train2_rat, test2_rat, result_as_dict=True)]

        # convert numpy to list for dict equal assertion
        result_list[0] = dict((user, list(filter_list)) for user, filter_list in result_list[0].items())
        result_list[1] = dict((user, list(filter_list)) for user, filter_list in result_list[1].items())

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

        self.assertTrue(len(expected_list), len(result_list))

        self.assertDictListCountEqual(expected_list[0], result_list[0])
        self.assertDictListCountEqual(expected_list[1], result_list[1])


class TestTrainingItemsMethodology(TestMethodology):

    def test_filter_all(self):

        train_1 = TrainingItemsMethodology()
        train_2 = TrainingItemsMethodology()

        train_1.setup(train1_rat, test1_rat)
        train_2.setup(train2_rat, test2_rat)

        result_list = [train_1.filter_all(train1_rat, test1_rat, ids_as_str=True),
                       train_2.filter_all(train2_rat, test2_rat, ids_as_str=True)]

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

        train_1 = TrainingItemsMethodology(only_greater_eq=3)
        train_2 = TrainingItemsMethodology(only_greater_eq=3)

        train_1.setup(train1_rat, test1_rat)
        train_2.setup(train2_rat, test2_rat)

        result_list = [train_1.filter_all(train1_rat, test1_rat),
                       train_2.filter_all(train2_rat, test2_rat)]

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

        train_1 = TrainingItemsMethodology()
        train_2 = TrainingItemsMethodology()

        train_1.setup(train1_rat, test1_rat)
        train_2.setup(train2_rat, test2_rat)

        result_list = [train_1.filter_all(train1_rat, test1_rat, result_as_dict=True),
                       train_2.filter_all(train2_rat, test2_rat, result_as_dict=True)]

        # convert numpy to list for dict equal assertion
        result_list[0] = dict((user, list(filter_list)) for user, filter_list in result_list[0].items())
        result_list[1] = dict((user, list(filter_list)) for user, filter_list in result_list[1].items())

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

        self.assertTrue(len(expected_list), len(result_list))

        self.assertDictListCountEqual(expected_list[0], result_list[0])
        self.assertDictListCountEqual(expected_list[1], result_list[1])


class TestAllItemsMethodology(TestMethodology):

    @classmethod
    def setUpClass(cls) -> None:
        cls.all_items = ["tt0112281",
                         "tt0112302",
                         "tt0112346",
                         "tt0112453",
                         "inew1",
                         "inew2",
                         "inew3",
                         "inew4"]

    def test_filter_all(self):
        all_1 = AllItemsMethodology()
        all_2 = AllItemsMethodology()

        all_1.setup(train1_rat, test1_rat)
        all_2.setup(train2_rat, test2_rat)

        result_list = [all_1.filter_all(train1_rat, test1_rat),
                       all_2.filter_all(train2_rat, test2_rat)]

        expected_split_1 = pd.DataFrame({
            'from_id': ["001", "001", "001", "001", "001", "001",
                        "002", "002", "002", "002", "002", "002", "002",
                        "003", "003", "003", "003", "003", "003", "003"],
            'to_id': ["tt0112346", "tt0112453", "tt0112641", "tt0112760", "tt0112896", "tt0113041",
                      "tt0112281", "tt0112302", "tt0112453", "tt0112641", "tt0112760", "tt0112896", "tt0113041",
                      "tt0112281", "tt0112302", "tt0112346", "tt0112641", "tt0112760", "tt0112896", "tt0113041"]
        })

        expected_split_2 = pd.DataFrame({
            'from_id': ["001", "001", "001", "001", "001", "001",
                        "002", "002", "002", "002", "002", "002",
                        "003", "003", "003", "003", "003", "003"],
            'to_id': ["tt0112896", "tt0113041", "tt0112281", "tt0112302", "tt0112346", "tt0112453",
                      "tt0112760", "tt0113041", "tt0112281", "tt0112302", "tt0112346", "tt0112453",
                      "tt0112641", "tt0112760", "tt0112896", "tt0112302", "tt0112346", "tt0112453"]
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
        all_1 = AllItemsMethodology()
        all_2 = AllItemsMethodology()

        all_1.setup(train1_rat, test1_rat)
        all_2.setup(train2_rat, test2_rat)

        result_list = [all_1.filter_all(train1_rat, test1_rat, result_as_dict=True),
                       all_2.filter_all(train2_rat, test2_rat, result_as_dict=True)]

        # convert numpy to list for dict equal assertion
        result_list[0] = dict((user, list(filter_list)) for user, filter_list in result_list[0].items())
        result_list[1] = dict((user, list(filter_list)) for user, filter_list in result_list[1].items())

        expected_split_1 = {
            "001": ["tt0112346", "tt0112453", "tt0112641", "tt0112760", "tt0112896", "tt0113041"],
            "002": ["tt0112281", "tt0112302", "tt0112453", "tt0112641", "tt0112760", "tt0112896", "tt0113041"],
            "003": ["tt0112281", "tt0112302", "tt0112346", "tt0112641", "tt0112760", "tt0112896", "tt0113041"]
        }

        expected_split_2 = {
            "001": ["tt0112896", "tt0113041", "tt0112281", "tt0112302", "tt0112346", "tt0112453"],
            "002": ["tt0112760", "tt0113041", "tt0112281", "tt0112302", "tt0112346", "tt0112453"],
            "003": ["tt0112641", "tt0112760", "tt0112896", "tt0112302", "tt0112346", "tt0112453"]
        }

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        self.assertDictListCountEqual(expected_list[0], result_list[0])
        self.assertDictListCountEqual(expected_list[1], result_list[1])

    def test_items_new(self):
        all_1 = AllItemsMethodology(items_list=self.all_items)
        all_2 = AllItemsMethodology(items_list=self.all_items)

        all_1.setup(train1_rat, test1_rat)
        all_2.setup(train2_rat, test2_rat)

        result_list = [all_1.filter_all(train1_rat, test1_rat, result_as_dict=True),
                       all_2.filter_all(train2_rat, test2_rat, result_as_dict=True)]

        # convert numpy to list for dict equal assertion
        result_list[0] = dict((user, list(filter_list)) for user, filter_list in result_list[0].items())
        result_list[1] = dict((user, list(filter_list)) for user, filter_list in result_list[1].items())

        expected_split_1 = {
            '001': ["tt0112346", "tt0112453", "inew1", "inew2", "inew3", "inew4"],
            '002': ["tt0112281", "tt0112302", "tt0112453", "inew1", "inew2", "inew3", "inew4"],
            '003': ["tt0112281", "tt0112302", "tt0112346", "inew1", "inew2", "inew3", "inew4"]
        }

        expected_split_2 = {
            '001': ["tt0112281", "tt0112302", "tt0112346", "tt0112453", "inew1", "inew2", "inew3", "inew4"],
            '002': ["tt0112281", "tt0112302", "tt0112346", "tt0112453", "inew1", "inew2", "inew3", "inew4"],
            '003': ["tt0112302", "tt0112346", "tt0112453", "inew1", "inew2", "inew3", "inew4"]
        }

        expected_list = [expected_split_1, expected_split_2]

        self.assertTrue(len(expected_list), len(result_list))

        self.assertDictListCountEqual(expected_list[0], result_list[0])
        self.assertDictListCountEqual(expected_list[1], result_list[1])


if __name__ == "__main__":
    unittest.main()

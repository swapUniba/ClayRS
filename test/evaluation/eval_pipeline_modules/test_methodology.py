from unittest import TestCase
import pandas as pd
import os
import numpy as np

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split
from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import TestRatingsMethodology, TestItemsMethodology, \
    TrainingItemsMethodology, AllItemsMethodology
from orange_cb_recsys.utils.const import root_path

contents_dir = os.path.join(root_path, 'contents')

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

split_list = [Split(train1, test1), Split(train2, test2)]


class TestMethodology(TestCase):

    def test_get_item_to_predict(self):

        result_list = TestRatingsMethodology().get_item_to_predict(split_list)

        # for every user get the items in its test_set1
        expected_list = [test1[['from_id', 'to_id']], test2[['from_id', 'to_id']]]

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):

            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

    def test_get_item_to_predict_only_greater_eq(self):

        result_list = TestRatingsMethodology(only_greater_eq=3).get_item_to_predict(split_list)

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


# poor choice of words sadly
class TestTestItemsMethodology(TestCase):

    def test_get_item_to_predict(self):

        result_list = TestItemsMethodology().get_item_to_predict(split_list)

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

    def test_get_item_to_predict_only_greater_eq(self):
        result_list = TestItemsMethodology(only_greater_eq=3).get_item_to_predict(split_list)

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


class TestTrainingItemsMethodology(TestCase):

    def test_get_item_to_predict(self):

        result_list = TrainingItemsMethodology().get_item_to_predict(split_list)

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

    def test_get_item_to_predict_only_greater_eq(self):
        result_list = TrainingItemsMethodology(only_greater_eq=3).get_item_to_predict(split_list)

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


class TestAllItemsMethodology(TestCase):

    def test_get_item_to_predict(self):

        movies_dir = os.path.join(contents_dir, 'movies_multiple_repr')

        all_items = [os.path.splitext(f)[0] for f in os.listdir(movies_dir)
                     if os.path.isfile(os.path.join(movies_dir, f)) and f.endswith('xz')]

        all_items = set(all_items)

        result_list = AllItemsMethodology(all_items).get_item_to_predict(split_list)

        expected_list = []

        # for every user get all items in 'all_items' except the items present in the training
        # set of the user
        for split in split_list:

            train = split.train
            test = split.test

            expected_split = {'from_id': [], 'to_id': []}

            for user in set(test['from_id']):
                # Extract all items rated by the user
                user_train = set(train.query('from_id == @user')['to_id'])

                # Get all items that are not in the train set of the user
                expected_for_user = [item for item in all_items if item not in user_train]

                expected_split['from_id'].extend([user for i in range(len(expected_for_user))])
                expected_split['to_id'].extend(expected_for_user)

            expected_list.append(pd.DataFrame(expected_split))

        self.assertTrue(len(expected_list), len(result_list))

        for expected, result in zip(expected_list, result_list):

            expected = np.array(expected)
            expected.sort(axis=0)

            result = np.array(result)
            result.sort(axis=0)

            self.assertTrue(np.array_equal(expected, result))

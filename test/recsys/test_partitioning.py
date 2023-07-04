import unittest
from unittest import TestCase
import pandas as pd
import numpy as np

from clayrs.content_analyzer.ratings_manager.ratings import Ratings
from clayrs.recsys.partitioning import HoldOutPartitioning, KFoldPartitioning, BootstrapPartitioning

original_ratings = pd.DataFrame.from_dict(
    {'from_id': ["001", "001", "001", "002", "002", "002", "003", "003", "003", "004", "004", "004"],
     'to_id': ["aaa", "bbb", "ccc", "aaa", "ddd", "ccc", "ccc", "aaa", "ddd", "bbb", "ddd", "ccc"],
     'rating': [0.8, 0.7, 0.3, -0.4, 1.0, 0.4, -0.2, 0.1, -0.3, 0.5, -0.9, 0.7]})

original_ratings = Ratings.from_dataframe(original_ratings)

only_002_can_split = pd.DataFrame.from_dict(
    {'from_id': ["001", "002", "002", "002", "003", "004"],
     'to_id': ["aaa", "aaa", "ddd", "ccc", "ccc", "aaa"],
     'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1]})

only_002_can_split = Ratings.from_dataframe(only_002_can_split)


class TestPartitioning(TestCase):

    def check_partition_correct(self, train, test, original):
        def arr_in_list(arr, list_of_arr):
            """
            Returns true if arr is in list_of_arr
            """
            return any(np.array_equal(arr, sub_arr, equal_nan=True) for sub_arr in list_of_arr)

        original_list = [row for row in original._uir]
        train_list = [row for row in train._uir]
        test_list = [row for row in test._uir]

        # Check that train and test are a partition
        # (must sort when checking for equality since they can have different ordering)
        train_not_in_test = [train_row for train_row in train_list if not arr_in_list(train_row, test_list)]
        np.testing.assert_array_equal(np.sort(train_list, axis=0), np.sort(train_not_in_test, axis=0))
        test_not_in_train = [test_row for test_row in test_list if not arr_in_list(test_row, train_list)]
        np.testing.assert_array_equal(np.sort(test_list, axis=0), np.sort(test_not_in_train, axis=0))

        # Check that the union of the two give the original data
        # (must sort when checking for equality since they can have different ordering)
        union_list = train_list + test_list
        np.testing.assert_array_equal(np.sort(original_list, axis=0), np.sort(union_list, axis=0))

    def test_split_all_str(self):

        # any partitioning technique will work, check that with user list of int and user list of str
        # we obtain same result
        ho = HoldOutPartitioning(random_state=42)

        [result_train_int], [result_test_int] = ho.split_all(original_ratings,
                                                             user_list=original_ratings.user_map[["001"]])
        [result_train_str], [result_test_str] = ho.split_all(original_ratings,
                                                             user_list={"001"})

        np.testing.assert_array_equal(result_train_int, result_train_str)
        np.testing.assert_array_equal(result_test_int, result_test_str)

class TestKFoldPartitioning(TestPartitioning):

    def test_split_all(self):

        kf = KFoldPartitioning(n_splits=2)

        result_train, result_test = kf.split_all(original_ratings)

        for train, test in zip(result_train, result_test):
            self.check_partition_correct(train, test, original_ratings)

        int_user_list = original_ratings.user_map[['001']]
        result_train, result_test = kf.split_all(original_ratings, user_list=set(int_user_list))

        for train, test in zip(result_train, result_test):
            only_001_train = train.unique_user_id_column
            only_001_test = test.unique_user_id_column

            self.assertTrue(len(only_001_train) == 1)
            self.assertTrue(len(only_001_test) == 1)

            self.assertIn("001", only_001_train)
            self.assertIn("001", only_001_test)

            int_user_list = original_ratings.user_map[['001']]
            original_001 = original_ratings.filter_ratings(user_list=int_user_list)

            self.check_partition_correct(train, test, original_001)

    def test_split_some_missing(self):
        kf = KFoldPartitioning(n_splits=3)

        int_user_list = original_ratings.user_map[['002']]
        result_train, result_test = kf.split_all(original_ratings, user_list=set(int_user_list))

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to perform 3 split
            only_002_train = train.unique_user_id_column
            only_002_test = test.unique_user_id_column

            self.assertTrue(len(only_002_train) == 1)
            self.assertTrue(len(only_002_test) == 1)

            self.assertIn("002", only_002_train)
            self.assertIn("002", only_002_test)

            int_user_list = original_ratings.user_map[['002']]
            original_002 = original_ratings.filter_ratings(user_list=int_user_list)

            self.check_partition_correct(train, test, original_002)

    def test_split_all_raise_error(self):
        kf = KFoldPartitioning(n_splits=3, skip_user_error=False)

        with self.assertRaises(ValueError):
            kf.split_all(only_002_can_split)


class TestHoldOutPartitioning(TestPartitioning):

    def test_split_all(self):
        train_instances = 1
        test_instances = 2

        ho = HoldOutPartitioning(train_set_size=train_instances, test_set_size=test_instances)

        result_train, result_test = ho.split_all(original_ratings)

        for train, test in zip(result_train, result_test):
            self.check_partition_correct(train, test, original_ratings)

            for user in original_ratings.unique_user_idx_column:

                len_train_user = len(train.get_user_interactions(user))
                len_test_user = len(test.get_user_interactions(user))

                self.assertEqual(train_instances, len_train_user)
                self.assertEqual(test_instances, len_test_user)

        int_user_list = original_ratings.user_map[['001']]
        result_train, result_test = ho.split_all(original_ratings, user_list=set(int_user_list))

        for train, test in zip(result_train, result_test):
            only_001_train = train.unique_user_id_column
            only_001_test = test.unique_user_id_column

            self.assertTrue(len(only_001_train) == 1)
            self.assertTrue(len(only_001_test) == 1)

            self.assertIn("001", only_001_train)
            self.assertIn("001", only_001_test)

            int_user_list = original_ratings.user_map[['001']]
            original_001 = original_ratings.filter_ratings(user_list=int_user_list)

            self.check_partition_correct(train, test, original_001)

    def test_split_some_missing(self):

        hold_percentage = 0.4

        ho = HoldOutPartitioning(train_set_size=hold_percentage)

        int_user_list = original_ratings.user_map[['002']]
        result_train, result_test = ho.split_all(original_ratings, user_list=set(int_user_list))

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to hold 0.4 as train
            only_002_train = train.unique_user_id_column
            only_002_test = test.unique_user_id_column

            self.assertTrue(len(only_002_train) == 1)
            self.assertTrue(len(only_002_test) == 1)

            self.assertIn('002', only_002_train)
            self.assertIn('002', only_002_test)

            int_user_list = original_ratings.user_map['002']
            original_002 = original_ratings.filter_ratings(user_list=int_user_list)

            self.check_partition_correct(train, test, original_002)

    def test_split_all_raise_error(self):

        hold_percentage = 0.3

        ho = HoldOutPartitioning(train_set_size=hold_percentage, skip_user_error=False)

        with self.assertRaises(ValueError):
            ho.split_all(original_ratings)

    def test__check_percentage(self):

        # train percentage bigger than 1
        train_percentage = 1.5
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=train_percentage)

        # negative train percentage
        train_percentage = -0.2
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=train_percentage)

        # test percentage greater than 1
        test_percentage = 1.5
        with self.assertRaises(ValueError):
            HoldOutPartitioning(test_set_size=test_percentage)

        # negative test percentage
        test_percentage = -0.2
        with self.assertRaises(ValueError):
            HoldOutPartitioning(test_set_size=test_percentage)

        # sum of train and test percentages is greater than 1
        train_percentage = 1.5
        test_percentage = 1.0
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=train_percentage, test_set_size=test_percentage)


class TestBootstrapPartitioning(TestPartitioning):

    def check_partition_correct(self, train, test, original):

        def arr_in_list(arr, list_of_arr):
            """
            Returns true if arr is in list_of_arr
            """
            return any(np.array_equal(arr, sub_arr, equal_nan=True) for sub_arr in list_of_arr)

        original_list = [row for row in original._uir]
        train_list = [row for row in train._uir]
        test_list = [row for row in test._uir]

        # Check that train and test are a partition
        # (must sort when checking for equality since they can have different ordering)
        train_not_in_test = [train_row for train_row in train_list if not arr_in_list(train_row, test_list)]
        np.testing.assert_array_equal(np.sort(train_list, axis=0), np.sort(train_not_in_test, axis=0))
        test_not_in_train = [test_row for test_row in test_list if not arr_in_list(test_row, train_list)]
        np.testing.assert_array_equal(np.sort(test_list, axis=0), np.sort(test_not_in_train, axis=0))

        # Check that the union of the two give the original data.
        # We remove any duplicate that can naturally happen due to the resampling of the
        # bootstrap method
        train_list_unique = []
        for row in train._uir:
            if not arr_in_list(row, train_list_unique):
                train_list_unique.append(row)

        union_list = train_list_unique + test_list

        np.testing.assert_array_equal(np.sort(original_list, axis=0), np.sort(union_list, axis=0))

    def test_split_all(self):
        bs = BootstrapPartitioning(random_state=5)

        [train], [test] = bs.split_all(original_ratings)

        # check that all users have been considered for the train set
        self.assertTrue("001" in train.user_id_column)
        self.assertTrue("002" in train.user_id_column)
        self.assertTrue("003" in train.user_id_column)
        self.assertTrue("004" in train.user_id_column)

        # check that all users have at least one duplicate in their train set
        # (with this particular random_state == 5 each user has 2 duplicates in the train set)
        for user_idx in train.unique_user_idx_column:

            user_unique_interactions = []
            user_interactions = train.get_user_interactions(user_idx)

            for user_interaction in user_interactions:
                # we stop (break) as soon as we find a duplicate
                if any(np.array_equal(user_interaction, unique, equal_nan=True) for unique in user_unique_interactions):
                    break
                user_unique_interactions.append(user_interaction)

            # means that a duplicate has been found
            self.assertTrue(len(user_interactions) != len(user_unique_interactions))

        # all users are present in the test set
        int_users_list = original_ratings.user_map[['001', '002', '003', '004']]
        original_002 = original_ratings.filter_ratings(int_users_list)

        self.check_partition_correct(train, test, original_002)

    def test_split_all_only_002(self):

        bs = BootstrapPartitioning(random_state=5)

        [train], [test] = bs.split_all(only_002_can_split)

        # 001, 003 and 004 had not enough ratings and so with this particular random state
        # the resampling will give us empty test set for those user, meaning that they will not be
        # present in the final train and test set
        self.assertTrue("001" not in train.user_id_column)
        self.assertTrue("002" in train.user_id_column)
        self.assertTrue("003" not in train.user_id_column)
        self.assertTrue("004" not in train.user_id_column)

        # only user 002 is present in train and test set
        int_user_list = only_002_can_split.user_map[['002']]
        original_002 = only_002_can_split.filter_ratings(int_user_list)
        self.check_partition_correct(train, test, original_002)

    def test_split_raise_error(self):
        bs = BootstrapPartitioning(random_state=5)

        # with this particular random state user 001 has not enough ratings and the resampling
        # will get all its ratings, making the test set empty, and so an error will be raised
        # by the split_single method
        int_user = only_002_can_split.user_map['001']
        user_001_rat = only_002_can_split.get_user_interactions(int_user)

        with self.assertRaises(ValueError):
            bs.split_single(user_001_rat)


if __name__ == '__main__':
    unittest.main()

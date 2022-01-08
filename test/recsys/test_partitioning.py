from unittest import TestCase
import pandas as pd

from orange_cb_recsys.recsys.partitioning import HoldOutPartitioning, KFoldPartitioning

original_frame = pd.DataFrame.from_dict(
    {'from_id': ["001", "001", "002", "002", "002", "003", "003", "004", "004"],
     'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "aaa", "ddd", "ccc"],
     'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.5, 0.7]})


class TestPartitioning(TestCase):
    def check_partition_correct(self, train, test, original):

        original_list = [list(row) for row in original.itertuples(index=False)]
        train_list = [list(row) for row in train.itertuples(index=False)]
        test_list = [list(row) for row in test.itertuples(index=False)]

        # Check that train and test are a partition
        train_not_in_test = [row for row in train_list if row not in test_list]
        self.assertCountEqual(train_list, train_not_in_test)  # Count so regardless of order
        test_not_in_train = [row for row in test_list if row not in train_list]
        self.assertCountEqual(test_list, test_not_in_train)  # Count so regardless of order

        # Check that the union of the two give the original data
        union_list = train_list + test_list
        self.assertCountEqual(original_list, union_list)  # Count so regardless of order


class TestKFoldPartitioning(TestPartitioning):

    def test_split_all(self):

        kf = KFoldPartitioning(n_splits=2)

        result_train, result_test = kf.split_all(original_frame)

        for train, test in zip(result_train, result_test):
            self.check_partition_correct(train, test, original_frame)

        result_train, result_test = kf.split_all(original_frame, user_id_list={'001'})

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to perform 3 split
            only_001_train = set(train['from_id'])
            only_001_test = set(train['from_id'])

            self.assertTrue(len(only_001_train) == 1)
            self.assertTrue(len(only_001_test) == 1)

            self.assertIn("001", only_001_train)
            self.assertIn("001", only_001_test)

            original_001 = original_frame[original_frame['from_id'] == "001"]

            self.check_partition_correct(train, test, original_001)

    def test_split_some_missing(self):
        kf = KFoldPartitioning(n_splits=3)

        result_train, result_test = kf.split_all(original_frame)

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to perform 3 split
            only_002_train = set(train['from_id'])
            only_002_test = set(train['from_id'])

            self.assertTrue(len(only_002_train) == 1)
            self.assertTrue(len(only_002_test) == 1)

            self.assertIn("002", only_002_train)
            self.assertIn("002", only_002_test)

            original_002 = original_frame[original_frame['from_id'] == "002"]

            self.check_partition_correct(train, test, original_002)

    def test_split_all_raise_error(self):
        kf = KFoldPartitioning(n_splits=3, skip_user_error=False)

        with self.assertRaises(ValueError):
            kf.split_all(original_frame)


class TestHoldOutPartitioning(TestPartitioning):

    def test_split_all(self):
        hold_percentage = 0.3

        ho = HoldOutPartitioning(train_set_size=hold_percentage)

        result_train, result_test = ho.split_all(original_frame)

        for train, test in zip(result_train, result_test):

            self.check_partition_correct(train, test, original_frame)

            train_percentage = (len(train) / len(original_frame))
            self.assertEqual(hold_percentage, train_percentage)

        result_train, result_test = ho.split_all(original_frame, user_id_list={'001'})

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to perform 3 split
            only_001_train = set(train['from_id'])
            only_001_test = set(train['from_id'])

            self.assertTrue(len(only_001_train) == 1)
            self.assertTrue(len(only_001_test) == 1)

            self.assertIn("001", only_001_train)
            self.assertIn("001", only_001_test)

            original_001 = original_frame[original_frame['from_id'] == "001"]

            self.check_partition_correct(train, test, original_001)

    def test_split_some_missing(self):

        hold_percentage = 0.4

        ho = HoldOutPartitioning(train_set_size=hold_percentage)

        result_train, result_test = ho.split_all(original_frame)

        for train, test in zip(result_train, result_test):
            # It's the only user for which is possible to perform 3 split
            only_002_train = set(train['from_id'])
            only_002_test = set(train['from_id'])

            self.assertTrue(len(only_002_train) == 1)
            self.assertTrue(len(only_002_test) == 1)

            self.assertIn("002", only_002_train)
            self.assertIn("002", only_002_test)

            original_002 = original_frame[original_frame['from_id'] == "002"]

            self.check_partition_correct(train, test, original_002)

    def test_split_all_raise_error(self):
        hold_percentage = 0.4

        ho = HoldOutPartitioning(train_set_size=hold_percentage, skip_user_error=False)

        with self.assertRaises(ValueError):
            ho.split_all(original_frame)

    def test__check_percentage(self):

        # Bigger than 1
        hold_percentage = 1.5
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=hold_percentage)

        # Negative percentage
        hold_percentage = -0.2
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=hold_percentage)

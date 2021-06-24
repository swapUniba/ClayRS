from unittest import TestCase
import pandas as pd

from orange_cb_recsys.evaluation.exceptions import PartitionError
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import HoldOutPartitioning, KFoldPartitioning

original_frame = pd.DataFrame.from_dict(
    {'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
     'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
     'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})


class TestKFoldPartitioning(TestCase):
    def test_iter(self):

        kf = KFoldPartitioning()

        kf.set_dataframe(original_frame)

        for train, test in kf:

            original_list = [list(row) for row in original_frame.itertuples(index=False)]
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

    def test__set_dataframe(self):
        empty_frame = pd.DataFrame()

        kf = KFoldPartitioning(n_splits=2)

        with self.assertRaises(PartitionError):
            kf.set_dataframe(empty_frame)


class TestHoldOutPartitioning(TestCase):
    def test_iter(self):

        hold_percentage = 0.5

        ho = HoldOutPartitioning(train_set_size=hold_percentage)

        ho.set_dataframe(original_frame)

        for train, test in ho:
            original_list = [list(row) for row in original_frame.itertuples(index=False)]
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

            train_percentage = (len(train) / len(original_frame))
            self.assertEqual(train_percentage, hold_percentage)

    def test__check_percentage(self):

        # Bigger than 1
        hold_percentage = 1.5
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=hold_percentage)

        # Negative percentage
        hold_percentage = -0.2
        with self.assertRaises(ValueError):
            HoldOutPartitioning(train_set_size=hold_percentage)

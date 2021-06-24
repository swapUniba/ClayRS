from unittest import TestCase
import pandas as pd

from orange_cb_recsys.evaluation.eval_model import PartitionModule
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import HoldOutPartitioning, KFoldPartitioning


class TestPartitionModule(TestCase):
    def test_split__single_kfold(self):
        user_ratings = pd.DataFrame.from_dict(
                        {'from_id': ["001", "001", "001", "001"],
                         'to_id': ["iphone", "ps4", "ps5", "xbox"],
                         'rating': [0.8, 0.7, -0.4, 1.0]})
        n_split = 2
        pm = PartitionModule(KFoldPartitioning(n_split))
        user_splits = pm._split_single(user_ratings)

        # No further tests since the partitioning technique is tested singularly
        self.assertEqual(len(user_splits), n_split)

    def test_split__single_hold_out(self):
        user_ratings = pd.DataFrame.from_dict(
                        {'from_id': ["001", "001", "001", "001"],
                         'to_id': ["iphone", "ps4", "ps5", "xbox"],
                         'rating': [0.8, 0.7, -0.4, 1.0]})

        pm = PartitionModule(HoldOutPartitioning())
        user_splits = pm._split_single(user_ratings)

        # No further tests since the partitioning technique is tested singularly
        self.assertEqual(len(user_splits), 1)

    def test_split_all_kfold(self):
        all_ratings = pd.DataFrame(
                        {'from_id': ["001", "001", "001", "001", "002", "002", "002", "003", "003"],
                         'to_id': ["iphone", "ps4", "ps5", "xbox", "realme", "airpods", "ps4", "beats", "dvd"],
                         'rating': [0.8, 0.7, -0.4, 1.0, 0.8, 0.7, -0.4, 1.0, 0.65]})
        n_split = 2
        pm = PartitionModule(KFoldPartitioning(n_split))
        split_list = pm.split_all(all_ratings, set(all_ratings.from_id))

        # No further tests since the partitioning technique is tested singularly
        self.assertEqual(len(split_list), n_split)

    def test_split_all_hold_out(self):
        all_ratings = pd.DataFrame(
                        {'from_id': ["001", "001", "001", "001", "002", "002", "002", "003", "003"],
                         'to_id': ["iphone", "ps4", "ps5", "xbox", "realme", "airpods", "ps4", "beats", "dvd"],
                         'rating': [0.8, 0.7, -0.4, 1.0, 0.8, 0.7, -0.4, 1.0, 0.65]})

        pm = PartitionModule(HoldOutPartitioning())
        splits = pm.split_all(all_ratings, set(all_ratings.from_id))

        # No further tests since the partitioning technique is tested singularly
        self.assertEqual(len(splits), 1)

    def test_all_skipping_user_exception(self):
        all_ratings = pd.DataFrame(
                        {'from_id': ["001", "001", "001", "001", "002", "002", "002", "003", "004", "004"],
                         'to_id': ["iphone", "ps4", "ps5", "xbox", "realme", "airpods", "ps4", "beats", "ps4", "ps5"],
                         'rating': [0.8, 0.7, -0.4, 1.0, 0.8, 0.7, -0.4, 1.0, 0.3, 0.6]})

        n_split = 2
        pm = PartitionModule(KFoldPartitioning(n_split))
        split_list = pm.split_all(all_ratings, set(all_ratings.from_id))

        # No further tests since the partitioning technique is tested singularly
        self.assertEqual(len(split_list), n_split)

        # Check that there are all users except 003 which is skipped since it has only 1 rating
        for split in split_list:
            self.assertIn('001', split.train['from_id'].values)
            self.assertIn('001', split.test['from_id'].values)
            self.assertIn('002', split.train['from_id'].values)
            self.assertIn('002', split.test['from_id'].values)
            self.assertNotIn('003', split.train['from_id'].values)
            self.assertNotIn('003', split.test['from_id'].values)
            self.assertIn('004', split.train['from_id'].values)
            self.assertIn('004', split.test['from_id'].values)

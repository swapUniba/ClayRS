import unittest
from collections import Counter
from unittest import TestCase
import pandas as pd

from clayrs.content_analyzer import Ratings
from clayrs.evaluation.utils import get_most_popular_items, pop_ratio_by_user, get_avg_pop, get_item_popularity


class TestUtils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        df = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u5', 'u5'],
            'to_id': ['i1', 'i2', 'i1', 'i50', 'i1', 'i2', 'i1', 'i50', 'i2', 'i70'],
            'score': [5, 2.5, 3, 4, 5, 1.2, 5, 4, 4.5, 3]
        })
        cls.custom_rat = Ratings.from_dataframe(df)

    def test_get_popularity(self):

        # n_frequency / n_users
        expected_pop_by_item = {'i1': 4 / 5, 'i2': 3 / 5, 'i50': 2 / 5, 'i70': 1 / 5}
        result_pop_by_item = get_item_popularity(self.custom_rat)

        self.assertCountEqual(expected_pop_by_item, result_pop_by_item)

    def test_popular_items(self):
        # there are 4 unique items, default percentage = 0.2, will return 1 most popular item
        pop_by_item = get_item_popularity(self.custom_rat)
        result = get_most_popular_items(pop_by_item)
        expected = {'i1'}

        self.assertEqual(expected, result)

        # there are 4 unique items, percentage = 0.5, will return 2 most popular item
        result = get_most_popular_items(pop_by_item, 0.5)
        expected = {'i1', 'i2'}

        self.assertEqual(expected, result)

    def test_pop_ratio_by_user(self):
        most_popular_items = {'i1'}
        result = pop_ratio_by_user(self.custom_rat, most_popular_items)

        # Expected popularity ratio is:
        # the number of item rated by the user that are in most_popular_items / n_rated items by the user

        expected = {'u1': 1/2, 'u2': 1/2, 'u3': 1/2, 'u4': 1/2, 'u5': 0}

        self.assertTrue(expected, result)

    def test_get_avg_pop(self):

        counter_popularity = Counter(self.custom_rat.item_id_column)

        # Expected result are item_popularity rated by user / n_item rated by user
        # item_popularity is the number of occurrences of the item in the 'to_id' column

        u1_idx = self.custom_rat.user_map["u1"]
        u1_items_idx = self.custom_rat.get_user_interactions(u1_idx)[:, 1].astype(int)
        u1_items_str = self.custom_rat.item_map[u1_items_idx]
        result_u1 = get_avg_pop(u1_items_str, counter_popularity)
        expected_u1 = (counter_popularity['i1'] + counter_popularity['i2']) / 2
        self.assertAlmostEqual(expected_u1, result_u1)

        u2_idx = self.custom_rat.user_map["u2"]
        u2_items_idx = self.custom_rat.get_user_interactions(u2_idx)[:, 1].astype(int)
        u2_items_str = self.custom_rat.item_map[u2_items_idx]
        result_u2 = get_avg_pop(u2_items_str, counter_popularity)
        expected_u2 = (counter_popularity['i1'] + counter_popularity['i50']) / 2
        self.assertAlmostEqual(expected_u2, result_u2)

        u3_idx = self.custom_rat.user_map["u3"]
        u3_items_idx = self.custom_rat.get_user_interactions(u3_idx)[:, 1].astype(int)
        u3_items_str = self.custom_rat.item_map[u3_items_idx]
        result_u3 = get_avg_pop(u3_items_str, counter_popularity)
        expected_u3 = (counter_popularity['i1'] + counter_popularity['i2']) / 2
        self.assertAlmostEqual(expected_u3, result_u3)

        u4_idx = self.custom_rat.user_map["u4"]
        u4_items_idx = self.custom_rat.get_user_interactions(u4_idx)[:, 1].astype(int)
        u4_items_str = self.custom_rat.item_map[u4_items_idx]
        result_u4 = get_avg_pop(u4_items_str, counter_popularity)
        expected_u4 = (counter_popularity['i1'] + counter_popularity['i50']) / 2
        self.assertAlmostEqual(expected_u4, result_u4)

        u5_idx = self.custom_rat.user_map["u5"]
        u5_items_idx = self.custom_rat.get_user_interactions(u5_idx)[:, 1].astype(int)
        u5_items_str = self.custom_rat.item_map[u5_items_idx]
        result_u5 = get_avg_pop(u5_items_str, counter_popularity)
        expected_u5 = (counter_popularity['i2'] + counter_popularity['i70']) / 2
        self.assertAlmostEqual(expected_u5, result_u5)


if __name__ == '__main__':
    unittest.main()

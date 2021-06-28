from collections import Counter
from unittest import TestCase
import pandas as pd
import numpy as np

from orange_cb_recsys.evaluation.utils import popular_items, pop_ratio_by_user, get_avg_pop


class TestUtils(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.DataFrame({
            'from_id': ['u1', 'u1', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4', 'u5', 'u5'],
            'to_id': ['i1', 'i2', 'i1', 'i50', 'i1', 'i2', 'i1', 'i50', 'i2', 'i70'],
            'score': [5, 2.5, 3, 4, 5, 1.2, 5, 4, 4.5, 3]
        })

    def test_popular_items(self):
        # there are 4 unique items, default percentage = 0.2, will return 1 most popular item
        result = popular_items(self.df)
        expected = {'i1'}

        self.assertEqual(expected, result)

        # there are 4 unique items, percentage = 0.5, will return 2 most popular item
        result = popular_items(self.df, 0.5)
        expected = {'i1', 'i2'}

        self.assertEqual(expected, result)

    def test_pop_ratio_by_user(self):
        most_popular_items = {'i1'}
        result = pop_ratio_by_user(self.df, most_popular_items)

        # Expected popularity ratio is:
        # the number of item rated by the user that are in most_popular_items / n_rated items by the user

        expected = {
            'from_id': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'popularity_ratio': [1/2, 1/2, 1/2, 1/2, 0]
        }

        expected = np.array(pd.DataFrame(expected))
        result = np.array(result)

        expected.sort(axis=0)
        result.sort(axis=0)

        self.assertTrue(np.array_equal(expected, result))

    def test_get_avg_pop(self):

        counter_popularity = Counter(self.df['to_id'])

        # Expected result are item_popularity rated by user / n_item rated by user
        # item_popularity is the number of occurrences of the item in the 'to_id' column

        u1_items = self.df.query('from_id == "u1"')['to_id']
        result_u1 = get_avg_pop(u1_items, counter_popularity)
        expected_u1 = (counter_popularity['i1'] + counter_popularity['i2']) / 2
        self.assertAlmostEqual(expected_u1, result_u1)

        u2_items = self.df.query('from_id == "u2"')['to_id']
        result_u2 = get_avg_pop(u2_items, counter_popularity)
        expected_u2 = (counter_popularity['i1'] + counter_popularity['i50']) / 2
        self.assertAlmostEqual(expected_u2, result_u2)

        u3_items = self.df.query('from_id == "u3"')['to_id']
        result_u3 = get_avg_pop(u3_items, counter_popularity)
        expected_u3 = (counter_popularity['i1'] + counter_popularity['i2']) / 2
        self.assertAlmostEqual(expected_u3, result_u3)

        u4_items = self.df.query('from_id == "u4"')['to_id']
        result_u4 = get_avg_pop(u4_items, counter_popularity)
        expected_u4 = (counter_popularity['i1'] + counter_popularity['i50']) / 2
        self.assertAlmostEqual(expected_u4, result_u4)

        u5_items = self.df.query('from_id == "u5"')['to_id']
        result_u5 = get_avg_pop(u5_items, counter_popularity)
        expected_u5 = (counter_popularity['i2'] + counter_popularity['i70']) / 2
        self.assertAlmostEqual(expected_u5, result_u5)

from unittest import TestCase
from orange_cb_recsys.utils.load_content import load_content_instance, remove_not_existent_items
import pandas as pd


class Test(TestCase):
    def test_load_content_instance(self):
        try:
            load_content_instance("aaa", '1')
        except FileNotFoundError:
            pass

    def test_remove_not_existent_items(self):
        ratings = pd.DataFrame({'to_id': ['tt0112281', 'aaaa']})
        remove_not_existent_items(ratings, 'contents/movielens_test1591885241.5520566')

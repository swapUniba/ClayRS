import os
from unittest import TestCase
from orange_cb_recsys.utils.load_content import load_content_instance, remove_not_existent_items, get_rated_items
import pandas as pd
from test import dir_test_files

movies_dir = os.path.join(dir_test_files, 'complex_contents', 'movies_codified/')


class Test(TestCase):
    def test_load_content_instance(self):
        self.assertIsNone(load_content_instance("not_existent", "invalid_item"))

    def test_remove_not_existent_items(self):
        ratings = pd.DataFrame({'to_id': ['tt0112281', 'aaaa']})
        result = remove_not_existent_items(ratings, movies_dir)

        self.assertNotIn('aaa', list(result.to_id))

    def test_get_rated_items(self):
        ratings = pd.DataFrame({'to_id': ['tt0112281', 'tt0113497']})
        loaded_items = get_rated_items(movies_dir, ratings)

        result_loaded_ids = [item.content_id for item in loaded_items]

        self.assertIn('tt0112281', result_loaded_ids)
        self.assertIn('tt0113497', result_loaded_ids)

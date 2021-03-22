import lzma
import os
import pickle
from unittest import TestCase
import pandas as pd

from orange_cb_recsys.recsys import IndexQuery


class TestIndexQuery(TestCase):
    def test_predict(self):

        ratings = pd.DataFrame.from_records([
            ("A000", "tt0112281", "sdfgd", 0.99, "54654675"),
            ("A000", "tt0112453", "sdfgd", 0, "54654675"),
            ("A000", "tt0112641", "sdfgd", 0.44, "54654675"),
            ("A000", "tt0112760", "sdfgd", -0.68, "54654675"),
            ("A000", "tt0112896", "sdfgd", -0.32, "54654675"),
            ("A000", "tt0113041", "sdfgd", 0.1, "54654675"),
            ("A000", "tt0113101", "sdfgd", -0.87, "54654675")
        ], columns=["from_id", "to_id", "original_rating", "score", "timestamp"])

        path = "../../../contents/movielens_test1591885241.5520566"
        items = []
        try:
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))
        except FileNotFoundError:
            path = "contents/movielens_test1591885241.5520566"
            file1 = os.path.join(path, "tt0114576.xz")
            with lzma.open(file1, "rb") as content_file:
                items.append(pickle.load(content_file))

            file2 = os.path.join(path, "tt0114709.xz")
            with lzma.open(file2, "rb") as content_file:
                items.append(pickle.load(content_file))

        t_index = IndexQuery(classic_similarity=False)

        ranking = t_index.predict(user_id='A000', ratings=ratings, recs_number=2, items_directory=path)

        self.assertEqual(2, len(ranking['to_id'].values))

        ranking = t_index.predict(user_id='A000', ratings=ratings, recs_number=2, items_directory=path,
                                  candidate_item_id_list=['tt0114576', 'tt0113987'])

        self.assertIn('tt0114576', ranking['to_id'].values)
        self.assertIn('tt0113987', ranking['to_id'].values)
        self.assertEqual(2, len(ranking['to_id'].values))

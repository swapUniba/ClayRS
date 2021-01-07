import lzma
import os
import pickle
from unittest import TestCase

from orange_cb_recsys.recsys import CosineSimilarity, CentroidVector, RecSysConfig, RecSys

import pandas as pd


class TestCentroidVector(TestCase):
    def test_predict(self):
        sim = CosineSimilarity()
        alg = CentroidVector('Plot', '1', sim)
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

        self.assertGreater(alg.predict('A000', ratings=ratings, recs_number=2, items_directory=path).rating[0], 0.9)

    def test_exceptions(self):
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

        try:
            t_centroid = CentroidVector(item_field='Title', field_representation='1', similarity=CosineSimilarity())
            t_centroid.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        except ValueError:
            pass

        try:
            t_centroid = CentroidVector(item_field='PLot', field_representation='4', similarity=CosineSimilarity())
            t_centroid.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        except ValueError:
            pass

        try:
            t_centroid = CentroidVector(item_field='PLot', field_representation='2', similarity=CosineSimilarity())
            t_centroid.predict('A000', ratings=ratings, recs_number=2, items_directory=path)
        except ValueError:
            pass


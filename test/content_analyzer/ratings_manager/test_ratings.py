import os
import shutil
import unittest
from unittest import TestCase

import pandas as pd
import numpy as np

from clayrs.content_analyzer.exceptions import UserNone, ItemNone
from clayrs.content_analyzer.ratings_manager.score_processor import NumberNormalizer
from clayrs.content_analyzer.ratings_manager.ratings import Ratings, StrIntMap
from clayrs.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, 'test_import_ratings.json')
raw_source = JSONFile(file_path)
raw_source_content = list(raw_source)


class TestStrIntMap(TestCase):

    def setUp(self) -> None:
        # test initialization with np.array as map
        self.array_map = np.array(["i1", "i2", "i3"])
        self.from_array_map = StrIntMap(self.array_map)

    def test_init(self):
        np.testing.assert_array_equal(self.array_map, self.from_array_map.map)

        # test initialization with a dict as map
        dict_map = {'i2': 1, "i3": 2, "i1": 0}
        from_dict_map = StrIntMap(dict_map)

        np.testing.assert_array_equal(self.array_map, from_dict_map.map)
        np.testing.assert_array_equal(self.from_array_map.map, from_dict_map.map)

        # test initialization with a StrIntMap
        new_strint_map = StrIntMap(from_dict_map)
        np.testing.assert_array_equal(from_dict_map.map, new_strint_map.map)

        # test initialization with an int array astype
        array_map = np.array([1, 2, 3])
        from_array_map = StrIntMap(array_map)

        self.assertTrue(np.issubdtype(from_array_map.map.dtype, str))

        # test initialization with a dict[int, int] as type
        dict_map = {10: 0, 11: 1, 12: 2}
        from_dict_map = StrIntMap(dict_map)

        self.assertTrue(np.issubdtype(from_dict_map.map.dtype, str))
        for expected, result in zip(dict_map, from_dict_map.map):

            self.assertEqual(str(expected), result)

    def test_init_exception(self):
        wrong_dict_map = {"i0": 10, "i1": 11, "i2": 12}

        with self.assertRaises(LookupError):
            StrIntMap(wrong_dict_map)

    def test_convert_int2str(self):
        conversion_result = self.from_array_map.convert_int2str(0)
        self.assertEqual("i1", conversion_result)

        conversion_result = self.from_array_map.convert_int2str(1)
        self.assertEqual("i2", conversion_result)

        conversion_result = self.from_array_map.convert_int2str(2)
        self.assertEqual("i3", conversion_result)

        # get last element
        conversion_result = self.from_array_map.convert_int2str(-1)
        self.assertEqual("i3", conversion_result)

        # going out of bounds
        with self.assertRaises(IndexError):
            self.from_array_map.convert_int2str(3)

    def test_convert_seq_int2str(self):
        conversion_result = self.from_array_map.convert_seq_int2str(np.array([0, 1, 2]))
        self.assertEqual(["i1", "i2", "i3"], list(conversion_result))

        # get second-last element, first element, last element
        conversion_result = self.from_array_map.convert_seq_int2str(np.array([-2, 0, -1]))
        self.assertEqual(["i2", "i1", "i3"], list(conversion_result))

        # convert empty sequence
        conversion_result = self.from_array_map.convert_seq_int2str(np.array([]))
        self.assertEqual([], list(conversion_result))

        # going out of bounds
        with self.assertRaises(IndexError):
            self.from_array_map.convert_seq_int2str(np.array([3, 4]))

    def test_convert_str2int(self):
        conversion_result = self.from_array_map.convert_str2int("i1")
        self.assertEqual(0, conversion_result)

        conversion_result = self.from_array_map.convert_str2int("i2")
        self.assertEqual(1, conversion_result)

        conversion_result = self.from_array_map.convert_str2int("i3")
        self.assertEqual(2, conversion_result)

        # going out of bounds
        with self.assertRaises(IndexError):
            self.from_array_map.convert_str2int("i10")

    def test_convert_seq_str2int(self):
        conversion_result = self.from_array_map.convert_seq_str2int(["i1", "i2", "i3"])
        self.assertEqual([0, 1, 2], list(conversion_result))

        conversion_result = self.from_array_map.convert_seq_str2int(["i2", "i1"])
        self.assertEqual([1, 0], list(conversion_result))

        # convert empty sequence
        conversion_result = self.from_array_map.convert_seq_str2int(np.array([]))
        self.assertEqual([], list(conversion_result))

        # going out of bounds
        with self.assertRaises(KeyError):
            self.from_array_map.convert_seq_str2int(np.array(["i3", "i4"]))

        with self.assertRaises(KeyError):
            self.from_array_map.convert_seq_str2int(np.array(["i4", "i5"]))

    def test_to_dict(self):
        expected = {"i1": 0, "i2": 1, "i3": 2}
        result = self.from_array_map.to_dict()

        self.assertDictEqual(expected, result)

        # test empty map
        empty_strint_map = StrIntMap(np.array([]))
        expected = {}
        result = empty_strint_map.to_dict()

        self.assertDictEqual(expected, result)

        # test append map
        empty_strint_map.append(["inew1", "inew2"])
        expected = {"inew1": 0, "inew2": 1}
        result = empty_strint_map.to_dict()

        self.assertDictEqual(expected, result)

    def test_getitem(self):
        # from str
        expected_index = 0
        res_index = self.from_array_map["i1"]
        self.assertEqual(expected_index, res_index)

        # from sequence of str
        expected_index = [1, 0]
        res_index = self.from_array_map[["i2", "i1"]]
        self.assertEqual(expected_index, list(res_index))

        # from array of str
        expected_index = [1, 0]
        res_index = self.from_array_map[np.array(["i2", "i1"])]
        self.assertEqual(expected_index, list(res_index))

        # from int
        expected_item = "i1"
        res_item = self.from_array_map[0]
        self.assertEqual(expected_item, res_item)

        # from sequence of int
        expected_item = ["i2", "i1"]
        res_item = self.from_array_map[[1, 0]]
        self.assertEqual(expected_item, list(res_item))

        # from array of int
        expected_item = ["i2", "i1"]
        res_item = self.from_array_map[np.array([1, 0])]
        self.assertEqual(expected_item, list(res_item))

    def test_exception_getitem(self):
        array_map = np.array(["i1", "i2", "i3"])
        from_array_map = StrIntMap(array_map)

        # passing a list with not supported elements (in this case bidimensional np array)
        with self.assertRaises(TypeError):
            res = from_array_map[np.array([[0, 1], [2, 3]])]

        # passing a list with not supported elements (in this case a pandas series)
        with self.assertRaises(TypeError):
            res = from_array_map[pd.Series(["i1", "i2"])]

        # passing a list with not supported elements (in this case None values)
        with self.assertRaises(TypeError):
            res = from_array_map[[None, None]]

        # single str not present
        with self.assertRaises(KeyError):
            res = from_array_map["missing"]

        # all list of str not present
        with self.assertRaises(KeyError):
            res = from_array_map[["missing_1", "missing_2"]]

        # some of list of str not present
        with self.assertRaises(KeyError):
            res = from_array_map[["missing", "i1"]]

        # single int not present
        with self.assertRaises(IndexError):
            res = from_array_map[99]

        # all list of int not present
        with self.assertRaises(IndexError):
            res = from_array_map[[99, 999]]

        # some of list of int not present
        with self.assertRaises(IndexError):
            res = from_array_map[[0, 5]]

    def test_iter(self):
        for el_original, el_map in zip(self.array_map, self.from_array_map):
            self.assertEqual(el_original, el_map)

    def test_eq(self):
        # test perfectly equal map
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})

        self.assertEqual(map1, map2)

        # test equal map with only different intial dict order
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i2": 1, "i3": 2, "i1": 0})

        self.assertEqual(map1, map2)

        # test different map disordered but with same elements
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i1": 1, "i2": 2, "i3": 0})

        self.assertNotEqual(map1, map2)

        # test completely different map but with same elements
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i10": 0, "i11": 1, "i12": 2})

        self.assertNotEqual(map1, map2)

        # test different map with only almost all elements in common
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i1": 0, "i2": 1, "i3": 2, "i4": 3})

        self.assertNotEqual(map1, map2)

        # test completely different map
        map1 = StrIntMap({"i1": 0, "i2": 1, "i3": 2})
        map2 = StrIntMap({"i11": 0, "i222": 1, "i33": 2, "i45": 3})

        self.assertNotEqual(map1, map2)

        # test empty map
        map1 = StrIntMap({})
        map2 = StrIntMap({})

        self.assertEqual(map1, map2)


class TestRatings(TestCase):

    # this tests user_id, user_idx, unique_user_id, unique_user_idx,
    # item_id, item_idx, unique_item_id, unique_item_idx
    # and score column (timestamp is missing)
    def _check_equal_expected_result(self, rat: Ratings):
        # ------------ check equality considering strings ------------
        self.assertTrue(len(rat) != 0)

        user_id_expected = [str(row['user_id']) for row in raw_source_content]
        item_id_expected = [str(row['item_id']) for row in raw_source_content]
        score_expected = [float(row['stars']) for row in raw_source_content]

        expected_ratings = [(user, item, score) for user, item, score in zip(user_id_expected,
                                                                             item_id_expected,
                                                                             score_expected)]

        user_id_result = rat.user_id_column
        item_id_result = rat.item_id_column
        score_result = rat.score_column

        result_ratings = [(user, item, score) for user, item, score in zip(user_id_result,
                                                                           item_id_result,
                                                                           score_result)]

        # check type is correct
        self.assertTrue(all(isinstance(user_id, str) for user_id in user_id_result))
        self.assertTrue(all(isinstance(item_id, str) for item_id in item_id_result))
        self.assertTrue(all(isinstance(score, float) for score in score_result))

        # check that expected is equal to result
        self.assertEqual(expected_ratings, result_ratings)

        # check that unique user ids and item ids are equal between expected and result
        self.assertCountEqual(list(set(user_id_expected)), list(rat.unique_user_id_column))
        self.assertCountEqual(list(set(item_id_expected)), list(rat.unique_item_id_column))

        # ------------ check equality considering integers ------------
        user_idx_expected = [rat.user_map[row['user_id']] for row in raw_source_content]
        item_idx_expected = [rat.item_map[row['item_id']] for row in raw_source_content]

        expected_ratings = [(user_idx, item_idx, score) for user_idx, item_idx, score in zip(user_idx_expected,
                                                                                             item_idx_expected,
                                                                                             score_expected)]

        user_idx_result = rat.user_idx_column
        item_idx_result = rat.item_idx_column

        result_ratings = [(user_idx, item_idx, score) for user_idx, item_idx, score in zip(user_idx_result,
                                                                                           item_idx_result,
                                                                                           score_result)]

        self.assertTrue(all(isinstance(user_idx, np.integer) for user_idx in user_idx_result))
        self.assertTrue(all(isinstance(item_idx, np.integer) for item_idx in item_idx_result))

        # check that unique user idxs and item idxs are equal between expected and result
        self.assertCountEqual(list(set(user_idx_expected)), list(rat.unique_user_idx_column))
        self.assertCountEqual(list(set(item_idx_expected)), list(rat.unique_item_idx_column))

        self.assertEqual(expected_ratings, result_ratings)

    def test_import_ratings_by_key(self):
        rat = Ratings(
            source=raw_source,
            user_id_column='user_id',
            item_id_column='item_id',
            score_column='stars')

        self._check_equal_expected_result(rat)

        self.assertTrue(len(rat.timestamp_column) == 0)

    def test_import_ratings_by_index(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4)

        self._check_equal_expected_result(rat)

        self.assertTrue(len(rat.timestamp_column) == 0)

    def test_import_ratings_w_timestamp_key(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            timestamp_column="timestamp"
        )

        self._check_equal_expected_result(rat)

        timestamp_expected = [int(row['timestamp']) for row in raw_source_content]
        timestamp_result = rat.timestamp_column

        self.assertTrue(np.issubdtype(timestamp_result.dtype, int))
        self.assertEqual(timestamp_expected, list(timestamp_result))

    def test_import_ratings_w_timestamp_index(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            timestamp_column=5
        )

        self._check_equal_expected_result(rat)

        timestamp_expected = [int(row['timestamp']) for row in raw_source_content]
        timestamp_result = rat.timestamp_column

        self.assertTrue(np.issubdtype(timestamp_result.dtype, int))
        self.assertEqual(timestamp_expected, list(timestamp_result))

    def test_import_ratings_w_score_processor(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            score_processor=NumberNormalizer(scale=(1, 5))
        )

        score_result = rat.score_column

        self.assertTrue(-1 <= score <= 1 for score in score_result)

    def test_import_ratings_w_custom_item_user_map(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            user_map={"01": 2, "02": 1, "03": 0},
            item_map={"a": 2, "b": 0, "c": 1}
        )

        # check user map
        expected_int_idxs = np.array([0, 1, 2])
        result_int_idxs = rat.user_map[["03", "02", "01"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["01", "02", "03"])
        result_str_ids = rat.user_map[[2, 1, 0]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

        # check item map
        expected_int_idxs = np.array([0, 1, 2])
        result_int_idxs = rat.item_map[["b", "c", "a"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["a", "b", "c"])
        result_str_ids = rat.item_map[[2, 0, 1]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

    def test_ratings_to_dataframe(self):
        ri = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4
        )

        expected_str_user_column = [row["user_id"] for row in raw_source_content]
        expected_str_item_column = [row["item_id"] for row in raw_source_content]
        expected_score_column = [float(row["stars"]) for row in raw_source_content]

        # convert to dataframe with original string ids
        result_df = ri.to_dataframe(ids_as_str=True)

        self.assertEqual(expected_str_user_column, list(result_df["user_id"]))
        self.assertEqual(expected_str_item_column, list(result_df["item_id"]))
        self.assertEqual(expected_score_column, list(result_df["score"]))

        # convert to dataframe with mapped integer idxs
        result_df = ri.to_dataframe(ids_as_str=False)

        expected_int_user_column = ri.user_map[expected_str_user_column]
        expected_int_item_column = ri.item_map[expected_str_item_column]

        np.testing.assert_array_equal(expected_int_user_column, result_df["user_id"])
        np.testing.assert_array_equal(expected_int_item_column, result_df["item_id"])

        # convert to dataframe with also timestamp column
        ri = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            timestamp_column="timestamp"
        )
        expected_timestamp_column = [int(row["timestamp"]) for row in raw_source_content]

        result_df = ri.to_dataframe(ids_as_str=True)

        self.assertEqual(expected_timestamp_column, list(result_df["timestamp"]))

    def test_ratings_to_csv(self):
        ri = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4
        )

        # Test save
        ri.to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame.csv'))

        # Test save first duplicate
        ri.to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame (1).csv'))

        # Test save second duplicate
        ri.to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame (2).csv'))

        # Test save with overwrite
        ri.to_csv('csv_test/', overwrite=True)
        self.assertTrue(os.path.isfile('csv_test/ratings_frame.csv'))
        self.assertFalse(os.path.isfile('csv_test/ratings_frame (3).csv'))

        # Test save with custom name
        ri.to_csv('csv_test/', 'ratings_custom_name')
        self.assertTrue(os.path.isfile('csv_test/ratings_custom_name.csv'))

        # remove test folder
        shutil.rmtree('csv_test/')

    def test_from_dataframe(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        # check ids as string
        expected_str_user_col = list(df_ratings["user_id"])
        expected_str_item_col = list(df_ratings["item_id"])

        self.assertEqual(expected_str_user_col, list(rat.user_id_column))
        self.assertEqual(expected_str_item_col, list(rat.item_id_column))

        # check idxs as int
        expected_int_user_col = rat.user_map[df_ratings['user_id'].to_list()]
        expected_int_item_col = rat.item_map[df_ratings['item_id'].to_list()]

        np.testing.assert_array_equal(expected_int_user_col, rat.user_idx_column)
        np.testing.assert_array_equal(expected_int_item_col, rat.item_idx_column)

        np.testing.assert_array_equal(df_ratings['score'], rat.score_column)

    def test_from_dataframe_w_custom_item_user_map(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings,
                                     item_map={"i2": 0, "i1": 1, "i80": 2},
                                     user_map={"u1": 2, "u2": 0, "u3": 1})

        # check user map
        expected_int_idxs = np.array([0, 1, 2])
        result_int_idxs = rat.user_map[["u2", "u3", "u1"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["u1", "u2", "u3"])
        result_str_ids = rat.user_map[[2, 0, 1]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

        # check item map
        expected_int_idxs = np.array([0, 1, 2])
        result_int_idxs = rat.item_map[["i2", "i1", "i80"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["i2", "i1", "i80"])
        result_str_ids = rat.item_map[[0, 1, 2]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

    def test_from_dataframe_w_None(self):
        # user column cannot contain None

        df_ratings = pd.DataFrame({
            'user_id': [None, 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1]
        })

        with self.assertRaises(UserNone):
            Ratings.from_dataframe(df_ratings)

        # item column cannot contain None

        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': [None, 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1]
        })

        with self.assertRaises(ItemNone):
            Ratings.from_dataframe(df_ratings)

        # None in score column will be converted to np.nan

        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [None, 3, 4, 1]
        })

        res = Ratings.from_dataframe(df_ratings)

        self.assertTrue(np.isnan(score) for score in res.score_column)

        # None in timestamp column will be converted to np.nan

        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1],
            'timestamp': [None, 300, 200, 100]
        })

        res = Ratings.from_dataframe(df_ratings)

        self.assertTrue(np.isnan(score) for score in res.score_column)

    def test_from_empty_dataframe(self):
        df_ratings = pd.DataFrame()

        rat = Ratings.from_dataframe(df_ratings)

        self.assertTrue(len(rat.user_id_column) == 0)
        self.assertTrue(len(rat.user_idx_column) == 0)
        self.assertTrue(len(rat.unique_user_idx_column) == 0)

        self.assertTrue(len(rat.item_id_column) == 0)
        self.assertTrue(len(rat.item_idx_column) == 0)
        self.assertTrue(len(rat.unique_item_idx_column) == 0)

        iter_expected = []
        iter_result = list(row for row in rat)

        self.assertEqual(iter_expected, iter_result)

    def test_exception_from_dataframe(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        # test column index as key not present in dataframe
        with self.assertRaises(KeyError):
            Ratings.from_dataframe(df_ratings, timestamp_column="not present")

        # test column index as int not present in dataframe
        with self.assertRaises(IndexError):
            Ratings.from_dataframe(df_ratings, timestamp_column=99)

        # test exception can't convert score column to float
        with self.assertRaises(ValueError):
            Ratings.from_dataframe(df_ratings, score_column="user_id")

        # test exception can't convert timestamp column to int
        with self.assertRaises(ValueError):
            Ratings.from_dataframe(df_ratings, timestamp_column="user_id")

    def test_from_list(self):
        list_no_timestamp = [('u1', 'i1', 5),
                             ('u1', 'i2', 4),
                             ('u2', 'i1', 2)]

        rat = Ratings.from_list(list_no_timestamp)

        # check ids as string columns
        self.assertEqual(['u1', 'u1', 'u2'], list(rat.user_id_column))
        self.assertEqual(['i1', 'i2', 'i1'], list(rat.item_id_column))

        # check idxs as int columns
        expected_int_users_col = rat.user_map[['u1', 'u1', 'u2']]
        expected_int_items_col = rat.item_map[['i1', 'i2', 'i1']]

        np.testing.assert_array_equal(expected_int_users_col, rat.user_idx_column)
        np.testing.assert_array_equal(expected_int_items_col, rat.item_idx_column)

        # test score and timestamp column is empty
        self.assertEqual([5.0, 4.0, 2.0], list(rat.score_column))
        self.assertTrue(len(rat.timestamp_column) == 0)

        list_w_timestamp = [('u1', 'i1', 5, 1),
                            ('u1', 'i2', 4, 2),
                            ('u2', 'i1', 2, 3)]

        rat = Ratings.from_list(list_w_timestamp)

        # check ids as string columns
        self.assertEqual(['u1', 'u1', 'u2'], list(rat.user_id_column))
        self.assertEqual(['i1', 'i2', 'i1'], list(rat.item_id_column))

        # check idxs as int columns
        expected_int_users_col = rat.user_map[['u1', 'u1', 'u2']]
        expected_int_items_col = rat.item_map[['i1', 'i2', 'i1']]

        np.testing.assert_array_equal(expected_int_users_col, rat.user_idx_column)
        np.testing.assert_array_equal(expected_int_items_col, rat.item_idx_column)

        # test score and timestamp column
        self.assertEqual([5.0, 4.0, 2.0], list(rat.score_column))
        self.assertEqual([1, 2, 3], list(rat.timestamp_column))

    def test_from_list_with_None(self):
        # user column can't contain None
        list_none_user = [(None, 'i1', 5),
                          ('u1', 'i2', 4),
                          ('u2', 'i1', 2)]

        with self.assertRaises(UserNone):
            Ratings.from_list(list_none_user)

        # item column can't contain None
        list_none_user = [('u1', 'i1', 5),
                          ('u1', None, 4),
                          ('u2', 'i1', 2)]

        with self.assertRaises(ItemNone):
            Ratings.from_list(list_none_user)

        list_none_score = [('u1', 'i1', 5),
                           ('u1', 'i2', None),
                           ('u2', 'i1', 2)]

        res = Ratings.from_list(list_none_score)

        # None scores are converted to np.nan
        self.assertTrue(np.isnan(score) for score in res.score_column)

        list_none_timestamp = [('u1', 'i1', 5, 1234),
                               ('u1', 'i2', 4, 1234),
                               ('u2', 'i1', 2, None)]

        res = Ratings.from_list(list_none_timestamp)

        # None timestamps are converted to np.nan
        self.assertTrue(np.isnan(score) for score in res.score_column)

    def test_from_list_w_custom_item_user_map(self):
        list_interactions = [('u1', 'i1', 5),
                             ('u1', 'i2', 4),
                             ('u2', 'i1', 2)]

        rat = Ratings.from_list(list_interactions,
                                user_map={"u2": 0, "u1": 1},
                                item_map={"i1": 0, "i2": 1})

        # check user map
        expected_int_idxs = np.array([0, 1])
        result_int_idxs = rat.user_map[["u2", "u1"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["u1", "u2"])
        result_str_ids = rat.user_map[[1, 0]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

        # check item map
        expected_int_idxs = np.array([0, 1])
        result_int_idxs = rat.item_map[["i1", "i2"]]

        np.testing.assert_array_equal(expected_int_idxs, result_int_idxs)

        expected_str_ids = np.array(["i2", "i1"])
        result_str_ids = rat.item_map[[1, 0]]

        np.testing.assert_array_equal(expected_str_ids, result_str_ids)

    def test_from_empty_list(self):
        list_interaction = []

        rat = Ratings.from_list(list_interaction)

        self.assertTrue(len(rat.user_id_column) == 0)
        self.assertTrue(len(rat.user_idx_column) == 0)
        self.assertTrue(len(rat.unique_user_idx_column) == 0)

        self.assertTrue(len(rat.item_id_column) == 0)
        self.assertTrue(len(rat.item_idx_column) == 0)
        self.assertTrue(len(rat.unique_item_idx_column) == 0)

        iter_expected = []
        iter_result = list(row for row in rat)

        self.assertEqual(iter_expected, iter_result)

    def test_from_list_exceptions(self):
        list_interactions_score_not_float = [('u1', 'i1', "not_float"),
                             ('u1', 'i2', 4),
                             ('u2', 'i1', 2)]

        # test exception can't convert score column to float
        with self.assertRaises(ValueError):
            Ratings.from_list(list_interactions_score_not_float)

        list_interactions_timestamp_not_int = [('u1', 'i1', 1, 1234),
                             ('u1', 'i2', 4, 456),
                             ('u2', 'i1', 2, "not_int")]

        # test exception can't convert timestamp column to int
        with self.assertRaises(ValueError):
            Ratings.from_list(list_interactions_timestamp_not_int)

    def test_from_uir(self):
        # user-item-rating matrix
        uir = [[0, 0, 2.0],
               [0, 1, 3.0],
               [1, 0, 4.0],
               [2, 2, 1.0]]

        uir = np.array(uir)

        # you also need to pass user map and item map since the user-item-rating matrix has already integers
        rat = Ratings.from_uir(uir, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        # check ids as string columns
        self.assertEqual(['u1', 'u1', 'u2', 'u3'], list(rat.user_id_column))
        self.assertEqual(['i1', 'i2', 'i1', 'i3'], list(rat.item_id_column))

        # check idxs as int columns
        expected_int_user_col = rat.user_map[['u1', 'u1', 'u2', 'u3']]
        expected_int_item_col = rat.item_map[['i1', 'i2', 'i1', 'i3']]

        np.testing.assert_array_equal(expected_int_user_col, rat.user_idx_column)
        np.testing.assert_array_equal(expected_int_item_col, rat.item_idx_column)

        # check score and timestamp empty
        self.assertEqual([2.0, 3.0, 4.0, 1.0], list(rat.score_column))
        self.assertTrue(len(rat.timestamp_column) == 0)

        # user-item-rating matrix with timestamp
        uir = [[0, 0, 2.0, 1234],
               [0, 1, 3.0, 1235],
               [1, 0, 4.0, 1236],
               [2, 2, 1.0, 1237]]

        uir = np.array(uir)

        # you also need to pass user map and item map since the user-item-rating matrix has already integers
        rat = Ratings.from_uir(uir, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        # check timestamp column
        self.assertEqual([1234, 1235, 1236, 1237], list(rat.timestamp_column))

    def test_from_empty_uir(self):
        # user-item-rating matrix
        uir = [[]]

        uir = np.array(uir)

        # you also need to pass user map and item map since the user-item-rating matrix has already integers
        rat = Ratings.from_uir(uir, {}, {})

        self.assertTrue(len(rat.user_id_column) == 0)
        self.assertTrue(len(rat.user_idx_column) == 0)
        self.assertTrue(len(rat.unique_user_idx_column) == 0)

        self.assertTrue(len(rat.item_id_column) == 0)
        self.assertTrue(len(rat.item_idx_column) == 0)
        self.assertTrue(len(rat.unique_item_idx_column) == 0)

        iter_expected = []
        iter_result = list(row for row in rat)

        self.assertEqual(iter_expected, iter_result)

    def test_from_uir_exception(self):
        # user-item-rating matrix but without the rating column
        uir = [[0, 0],
               [0, 1],
               [1, 0],
               [2, 2]]

        uir = np.array(uir)

        # uir should at least have 3 columns
        with self.assertRaises(ValueError):
            Ratings.from_uir(uir, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        # user-item-rating matrix but with strings
        uir = [["u1", "i1", 2.0],
               ["u1", "i2", 3.0],
               ["u2", "i1", 4.0],
               ["u3", "i3", 1.0]]

        uir = np.array(uir)

        # uir should contain numbers not strings
        with self.assertRaises(TypeError):
            Ratings.from_uir(uir, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

    def test_get_user_interaction(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        user_interactions = rat.get_user_interactions(rat.user_map['u1'])

        np.testing.assert_array_equal(rat.item_map[['i1', 'i2', 'i3']], user_interactions[:, 1])
        np.testing.assert_array_equal(np.array([2.0, 3.0, 4.0]), user_interactions[:, 2])

        # get first 2 user interactions
        user_interactions = rat.get_user_interactions(rat.user_map['u1'], head=2)

        np.testing.assert_array_equal(rat.item_map[['i1', 'i2']], user_interactions[:, 1])
        np.testing.assert_array_equal(np.array([2.0, 3.0]), user_interactions[:, 2])

    def test_filter_ratings(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        rat_filtered = rat.filter_ratings(rat.user_map[['u1', 'u3']])

        self.assertEqual(['u1', 'u1', 'u3'], list(rat_filtered.user_id_column))
        self.assertEqual(['i1', 'i2', 'i80'], list(rat_filtered.item_id_column))
        self.assertEqual([2.0, 3.0, 1.0], list(rat_filtered.score_column))

    def test_take_head_all(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u1', 'u4'],
            'item_id': ['i1', 'i2', 'i3', 'i80', 'i81', 'i82', 'i83', 'i84'],
            'score': [2, 3, 4, 1, 1, 2, 3, 5]
        })

        rat = Ratings.from_dataframe(df_ratings)

        # take first 2 elements for all users
        head_rat = rat.take_head_all(2)

        self.assertEqual(['u1', 'u1', 'u2', 'u2', 'u3', 'u4'], list(head_rat.user_id_column))
        self.assertEqual(['i1', 'i2', 'i3', 'i80', 'i82', 'i84'], list(head_rat.item_id_column))
        self.assertEqual([2.0, 3.0, 4.0, 1.0, 2.0, 5.0], list(head_rat.score_column))

    def test_exception_import_ratings(self):
        # Test exception column name not present in raw source
        with self.assertRaises(KeyError):
            Ratings(
                source=raw_source,
                user_id_column='not_existent',
                item_id_column='item_id',
                score_column='stars')

        # Test exception column index not present in raw source
        with self.assertRaises(IndexError):
            Ratings(
                source=raw_source,
                user_id_column=99,
                item_id_column='item_id',
                score_column='stars')

        # Test exception score column can't be converted into float
        with self.assertRaises(ValueError):
            Ratings(
                source=raw_source,
                user_id_column='user_id',
                item_id_column='item_id',
                score_column='review_title')

        # Test exception timestamp column can't be converted into int
        with self.assertRaises(ValueError):
            Ratings(
                source=raw_source,
                user_id_column='user_id',
                item_id_column='item_id',
                score_column='stars',
                timestamp_column='review_title'
            )

    def test_iter(self):
        # user-item-rating matrix
        uir = [[0, 0, 2.0],
               [0, 1, 3.0],
               [1, 0, 4.0],
               [2, 2, 1.0]]

        uir = np.array(uir)

        # you also need to pass user map and item map since the user-item-rating matrix has already integers
        rat = Ratings.from_uir(uir, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        for expected_row, result_row in zip(uir, rat):
            np.array_equal(expected_row, result_row)

    def test_eq(self):
        # test equal ratings
        uir1 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        uir2 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertEqual(rat1, rat2)

        # test equal ratings with timestamp
        uir1 = [[0, 0, 2.0, 123],
                [0, 1, 3.0, 124],
                [1, 0, 4.0, 125],
                [2, 2, 1.0, 126]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        uir2 = [[0, 0, 2.0, 123],
                [0, 1, 3.0, 124],
                [1, 0, 4.0, 125],
                [2, 2, 1.0, 126]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertEqual(rat1, rat2)

        # test different ratings where rows are in different order
        uir1 = [[0, 1, 3.0],
                [0, 0, 2.0],
                [2, 2, 1.0],
                [1, 0, 4.0]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        uir2 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertNotEqual(rat1, rat2)

        # test different ratings where both mappings are different
        uir1 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 1, 'u2': 2, 'u3': 0}, {'i1': 1, 'i2': 2, 'i3': 0})

        uir2 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertNotEqual(rat1, rat2)

        # test different ratings where only user map is different
        uir1 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 1, 'u2': 2, 'u3': 0}, {'i1': 0, 'i2': 1, 'i3': 2})

        uir2 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertNotEqual(rat1, rat2)

        # test different ratings where only item map is different
        uir1 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir1 = np.array(uir1)
        rat1 = Ratings.from_uir(uir1, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 1, 'i2': 2, 'i3': 0})

        uir2 = [[0, 0, 2.0],
                [0, 1, 3.0],
                [1, 0, 4.0],
                [2, 2, 1.0]]
        uir2 = np.array(uir2)
        rat2 = Ratings.from_uir(uir2, {'u1': 0, 'u2': 1, 'u3': 2}, {'i1': 0, 'i2': 1, 'i3': 2})

        self.assertNotEqual(rat1, rat2)


if __name__ == '__main__':
    unittest.main()

import os
import shutil
import unittest
from unittest import TestCase

import pandas as pd

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import Ratings, Interaction
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from test import dir_test_files

file_path = os.path.join(dir_test_files, 'test_import_ratings.json')
raw_source = JSONFile(file_path)
raw_source_content = list(raw_source)


class TestInteraction(TestCase):
    def test_all(self):
        interaction_no_timestamp = Interaction('u1', 'i1', 5)

        self.assertTrue(interaction_no_timestamp.user_id == 'u1')
        self.assertTrue(interaction_no_timestamp.item_id == 'i1')
        self.assertTrue(interaction_no_timestamp.score == 5)
        self.assertIsNone(interaction_no_timestamp.timestamp)

        interaction_w_timestamp = Interaction('u2', 'i2', 4, 'timestamp')

        self.assertTrue(interaction_w_timestamp.user_id == 'u2')
        self.assertTrue(interaction_w_timestamp.item_id == 'i2')
        self.assertTrue(interaction_w_timestamp.score == 4)
        self.assertTrue(interaction_w_timestamp.timestamp == 'timestamp')


class TestRatings(TestCase):

    # this only tests user, item and score column
    def _check_equal_expected_result(self, rat: Ratings):
        self.assertTrue(len(rat) != 0)

        user_id_expected = [str(row['user_id']) for row in raw_source_content]
        item_id_expected = [str(row['item_id']) for row in raw_source_content]
        score_expected = [float(row['stars']) for row in raw_source_content]

        user_id_result = rat.user_id_column
        item_id_result = rat.item_id_column
        score_result = rat.score_column

        self.assertTrue(all(isinstance(user_id, str) for user_id in user_id_result))
        self.assertTrue(all(isinstance(item_id, str) for item_id in item_id_result))
        self.assertTrue(all(isinstance(score, float) for score in score_result))

        self.assertEqual(user_id_expected, user_id_result)
        self.assertEqual(item_id_expected, item_id_result)
        self.assertEqual(score_expected, score_result)

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

        timestamp_expected = [row['timestamp'] for row in raw_source_content]

        timestamp_result = rat.timestamp_column

        self.assertTrue(all(isinstance(timestamp, str) for timestamp in timestamp_result))
        self.assertEqual(timestamp_expected, timestamp_result)

    def test_import_ratings_w_timestamp_index(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            timestamp_column=5
        )

        self._check_equal_expected_result(rat)

        timestamp_expected = [row['timestamp'] for row in raw_source_content]

        timestamp_result = rat.timestamp_column

        self.assertTrue(all(isinstance(timestamp, str) for timestamp in timestamp_result))
        self.assertEqual(timestamp_expected, timestamp_result)

    def test_import_ratings_w_score_processor(self):
        rat = Ratings(
            source=raw_source,
            user_id_column=0,
            item_id_column=1,
            score_column=4,
            score_processor=NumberNormalizer()
        )

        score_result = rat.score_column

        self.assertTrue(-1 <= score <= 1 for score in score_result)

    # def test_add_score_column(self):
    #     ri = Ratings(
    #         source=raw_source,
    #         user_id_column=0,
    #         item_id_column=1,
    #         score_column=4
    #     )
    # 
    #     expected_columns = ['user_id', 'item_id', 'score']
    #     result_columns = ri.columns
    # 
    #     self.assertEqual(expected_columns, result_columns)
    # 
    #     ri.add_score_column('stars', column_name='score_duplicate')
    # 
    #     expected_columns = ['user_id', 'item_id', 'score', 'score_duplicate']
    #     result_columns = ri.columns
    # 
    #     self.assertEqual(expected_columns, result_columns)
    # 
    #     expected = [float(row['stars']) for row in raw_source_content]
    # 
    #     score_column_added_frame = list(ri.frame['score_duplicate'])
    #     score_column_added_dict = [rating_tuple[2] for user_value in ri.dict.values() for rating_tuple in user_value]
    # 
    #     self.assertEqual(expected, score_column_added_frame)
    #     self.assertEqual(expected, score_column_added_dict)
    # 
    # def test_add_score_column_w_score_processor(self):
    #     ri = Ratings(
    #         source=raw_source,
    #         user_id_column=0,
    #         item_id_column=1,
    #         score_column=4
    #     )
    # 
    #     ratings = ri.frame
    # 
    #     expected_columns = ['user_id', 'item_id', 'score']
    #     result_columns = ri.columns
    # 
    #     self.assertEqual(expected_columns, result_columns)
    # 
    #     ri.add_score_column('review_title', column_name='text_polarity',
    #                         score_processor=TextBlobSentimentAnalysis())
    # 
    #     expected_columns = ['user_id', 'item_id', 'score', 'text_polarity']
    #     result_columns = ri.columns
    # 
    #     self.assertEqual(expected_columns, result_columns)
    # 
    #     score_column_added = list(ratings['text_polarity'])
    #     score_column_added_dict = [rating_tuple[2] for user_value in ri.dict.values() for rating_tuple in user_value]
    # 
    #     self.assertTrue(-1 <= score <= 1 for score in score_column_added)
    #     self.assertTrue(-1 <= score <= 1 for score in score_column_added_dict)

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

    def test_from_to_dataframe(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i1', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        self.assertEqual(rat.user_id_column, list(df_ratings['user_id']))
        self.assertEqual(rat.item_id_column, list(df_ratings['item_id']))
        self.assertEqual(rat.score_column, list(df_ratings['score']))

        df_result = rat.to_dataframe()

        self.assertEqual(list(df_ratings['user_id']), list(df_result['user_id']))
        self.assertEqual(list(df_ratings['item_id']), list(df_result['item_id']))
        self.assertEqual(list(df_ratings['score']), list(df_result['score']))

        # raise exception when a column doesn't exist in the original df by key
        with self.assertRaises(KeyError):
            Ratings.from_dataframe(df_ratings, user_column='not existent')

        # raise exception when a column doesn't exist in the original df by index
        with self.assertRaises(IndexError):
            Ratings.from_dataframe(df_ratings, user_column=10)

    def test_from_list(self):
        list_no_timestamp = [Interaction('u1', 'i1', 5),
                             Interaction('u1', 'i2', 4),
                             Interaction('u2', 'i1', 2)]

        rat = Ratings.from_list(list_no_timestamp)

        self.assertEqual(rat.user_id_column, ['u1', 'u1', 'u2'])
        self.assertEqual(rat.item_id_column, ['i1', 'i2', 'i1'])
        self.assertEqual(rat.score_column, [5, 4, 2])
        self.assertTrue(len(rat.timestamp_column) == 0)

        list_w_timestamp = [Interaction('u1', 'i1', 5, 'timestamp1'),
                            Interaction('u1', 'i2', 4, 'timestamp2'),
                            Interaction('u2', 'i1', 2, 'timestamp3')]

        rat = Ratings.from_list(list_w_timestamp)

        self.assertEqual(rat.user_id_column, ['u1', 'u1', 'u2'])
        self.assertEqual(rat.item_id_column, ['i1', 'i2', 'i1'])
        self.assertEqual(rat.score_column, [5, 4, 2])
        self.assertEqual(rat.timestamp_column, ['timestamp1', 'timestamp2', 'timestamp3'])

    def test_from_dict(self):
        dict_no_timestamp = {'u1': [Interaction('u1', 'i1', 5),
                                    Interaction('u1', 'i2', 4)],
                             'u2': [Interaction('u2', 'i1', 2)]}

        rat = Ratings.from_dict(dict_no_timestamp)

        self.assertEqual(rat.user_id_column, ['u1', 'u1', 'u2'])
        self.assertEqual(rat.item_id_column, ['i1', 'i2', 'i1'])
        self.assertEqual(rat.score_column, [5, 4, 2])
        self.assertTrue(len(rat.timestamp_column) == 0)

        dict_w_timestamp = {'u1': [Interaction('u1', 'i1', 5, 'timestamp1'),
                                   Interaction('u1', 'i2', 4, 'timestamp2')],
                            'u2': [Interaction('u2', 'i1', 2, 'timestamp3')]}

        rat = Ratings.from_dict(dict_w_timestamp)

        self.assertEqual(rat.user_id_column, ['u1', 'u1', 'u2'])
        self.assertEqual(rat.item_id_column, ['i1', 'i2', 'i1'])
        self.assertEqual(rat.score_column, [5, 4, 2])
        self.assertEqual(rat.timestamp_column, ['timestamp1', 'timestamp2', 'timestamp3'])

    def test_get_user_interaction(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u1', 'u2'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        user_interactions = rat.get_user_interactions('u1')

        self.assertEqual(['i1', 'i2', 'i3'], [interaction.item_id for interaction in user_interactions])
        self.assertEqual([2, 3, 4], [interaction.score for interaction in user_interactions])

        # get first 2 user interactions
        user_interactions = rat.get_user_interactions('u1', head=2)

        self.assertEqual(['i1', 'i2'], [interaction.item_id for interaction in user_interactions])
        self.assertEqual([2, 3], [interaction.score for interaction in user_interactions])

    def test_filter_ratings(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u3'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        rat_filtered = rat.filter_ratings(['u1', 'u3'])

        self.assertEqual(['u1', 'u1', 'u3'], rat_filtered.user_id_column)
        self.assertEqual(['i1', 'i2', 'i80'], rat_filtered.item_id_column)
        self.assertEqual([2, 3, 1], rat_filtered.score_column)

    def take_head_all(self):
        df_ratings = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'item_id': ['i1', 'i2', 'i3', 'i80'],
            'score': [2, 3, 4, 1]
        })

        rat = Ratings.from_dataframe(df_ratings)

        # take first element for all users
        head_rat = rat.take_head_all(1)

        self.assertEqual(['u1', 'u2'], head_rat.user_id_column)
        self.assertEqual(['i1', 'i3'], head_rat.item_id_column)
        self.assertEqual([2, 3], head_rat.score_column)

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

    # def test_exception_add_score_column(self):
    #     # Test exception score column can't be converted into float
    #     ri = Ratings(
    #         source=raw_source,
    #         user_id_column='user_id',
    #         item_id_column='item_id',
    #         score_column='stars')
    #
    #     with self.assertRaises(ValueError):
    #         ri.add_score_column('review_title', 'text')


if __name__ == '__main__':
    unittest.main()

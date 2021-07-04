import os
from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.utils.const import datasets_path


class TestRatingsImporter(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        file_path = os.path.join(datasets_path, 'test_import_ratings.json')

        cls.raw_source = JSONFile(file_path)

        cls.raw_source_content = list(cls.raw_source)

    def test_import_ratings_by_key(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column='user_id',
            to_id_column='item_id',
            score_column='stars')

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        from_id_result = list(ratings['from_id'])
        to_id_result = list(ratings['to_id'])
        score_result = list(ratings['score'])

        from_id_expected = [row['user_id'] for row in self.raw_source_content]
        to_id_expected = [row['item_id'] for row in self.raw_source_content]
        score_expected = [float(row['stars']) for row in self.raw_source_content]

        self.assertTrue(all(isinstance(from_id, str) for from_id in from_id_result))
        self.assertTrue(all(isinstance(to_id, str) for to_id in to_id_result))
        self.assertTrue(all(isinstance(score, float) for score in score_result))

        self.assertEqual(from_id_expected, from_id_result)
        self.assertEqual(to_id_expected, to_id_result)
        self.assertEqual(score_expected, score_result)

    def test_import_ratings_by_index(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4)

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        from_id_result = list(ratings['from_id'])
        to_id_result = list(ratings['to_id'])
        score_result = list(ratings['score'])

        from_id_expected = [row['user_id'] for row in self.raw_source_content]
        to_id_expected = [row['item_id'] for row in self.raw_source_content]
        score_expected = [float(row['stars']) for row in self.raw_source_content]

        self.assertTrue(all(isinstance(from_id, str) for from_id in from_id_result))
        self.assertTrue(all(isinstance(to_id, str) for to_id in to_id_result))
        self.assertTrue(all(isinstance(score, float) for score in score_result))

        self.assertEqual(from_id_expected, from_id_result)
        self.assertEqual(to_id_expected, to_id_result)
        self.assertEqual(score_expected, score_result)

    def test_import_ratings_w_timestamp_index(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4,
            timestamp_column=5
        )

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score', 'timestamp']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        timestamp_result = list(ratings['timestamp'])
        timestamp_expected = [row['timestamp'] for row in self.raw_source_content]

        self.assertTrue(all(isinstance(timestamp, str) for timestamp in timestamp_result))
        self.assertEqual(timestamp_expected, timestamp_result)

    def test_import_ratings_w_timestamp_key(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4,
            timestamp_column="timestamp"
        )

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score', 'timestamp']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        timestamp_result = list(ratings['timestamp'])
        timestamp_expected = [row['timestamp'] for row in self.raw_source_content]

        self.assertTrue(all(isinstance(timestamp, str) for timestamp in timestamp_result))
        self.assertEqual(timestamp_expected, timestamp_result)

    def test_import_ratings_w_score_processor(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4,
            score_processor=NumberNormalizer()
        )

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        score_result = list(ratings['score'])

        self.assertTrue(-1 <= score <= 1 for score in score_result)

    def test_add_score_column(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4
        )

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        ratings_w_added_column = ri.add_score_column('stars', column_name='score_duplicate')

        expected_columns = ['from_id', 'to_id', 'score', 'score_duplicate']
        result_columns = list(ratings_w_added_column.columns)

        self.assertEqual(expected_columns, result_columns)

        score_column_added = list(ratings['score_duplicate'])
        expected = [float(row['stars']) for row in self.raw_source_content]

        self.assertEqual(expected, score_column_added)

    def test_add_score_column_w_score_processor(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4
        )

        ratings = ri.import_ratings()

        expected_columns = ['from_id', 'to_id', 'score']
        result_columns = list(ratings.columns)

        self.assertEqual(expected_columns, result_columns)

        ratings_w_added_column = ri.add_score_column('review_title', column_name='text_polarity',
                                                     score_processor=TextBlobSentimentAnalysis())

        expected_columns = ['from_id', 'to_id', 'score', 'text_polarity']
        result_columns = list(ratings_w_added_column.columns)

        self.assertEqual(expected_columns, result_columns)

        score_column_added = list(ratings['text_polarity'])

        self.assertTrue(-1 <= score <= 1 for score in score_column_added)

    def test_ratings_to_csv(self):
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=0,
            to_id_column=1,
            score_column=4
        )

        ri.import_ratings()

        # Test save
        ri.imported_ratings_to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame.csv'))

        # Test save first duplicate
        ri.imported_ratings_to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame (1).csv'))

        # Test save second duplicate
        ri.imported_ratings_to_csv('csv_test/')
        self.assertTrue(os.path.isfile('csv_test/ratings_frame (2).csv'))

        # Test save with overwrite
        ri.imported_ratings_to_csv('csv_test/', overwrite=True)
        self.assertTrue(os.path.isfile('csv_test/ratings_frame.csv'))
        self.assertFalse(os.path.isfile('csv_test/ratings_frame (3).csv'))

        # Test save with custom name
        ri.imported_ratings_to_csv('csv_test/', 'ratings_custom_name')
        self.assertTrue(os.path.isfile('csv_test/ratings_custom_name.csv'))

    def test_exception_import_ratings(self):

        # Test exception column name not present in raw source
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column='not_existent',
            to_id_column='item_id',
            score_column='stars')

        with self.assertRaises(KeyError):
            ri.import_ratings()

        # Test exception column index not present in raw source
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column=99,
            to_id_column='item_id',
            score_column='stars')

        with self.assertRaises(IndexError):
            ri.import_ratings()

        # Test exception score column can't be converted into float
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column='user_id',
            to_id_column='item_id',
            score_column='review_title')

        with self.assertRaises(ValueError):
            ri.import_ratings()

    def test_exception_add_score_column(self):
        # Test exception score column can't be converted into float
        ri = RatingsImporter(
            source=self.raw_source,
            from_id_column='user_id',
            to_id_column='item_id',
            score_column='stars')

        with self.assertRaises(ValueError):
            ri.add_score_column('review_title', 'text')

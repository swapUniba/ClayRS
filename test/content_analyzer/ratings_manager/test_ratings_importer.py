from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter, RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestRatingsImporter(TestCase):
    def test_import_ratings(self):
        file_path = '../../../datasets/test_import_ratings.json'
        try:
            with open(file_path):
                pass
        except FileNotFoundError:
            file_path = 'datasets/test_import_ratings.json'

        RatingsImporter(source=JSONFile(file_path=file_path),
                        output_directory="test_ratings",
                        rating_configs=[
                            RatingsFieldConfig(field_name="review_title",
                                               processor=TextBlobSentimentAnalysis()),
                            RatingsFieldConfig(field_name="text",
                                               processor=TextBlobSentimentAnalysis()),
                            RatingsFieldConfig(field_name="stars",
                                               processor=NumberNormalizer(min_=0, max_=5))],
                        from_field_name="user_id",
                        to_field_name="item_id",
                        timestamp_field_name="timestamp").import_ratings()

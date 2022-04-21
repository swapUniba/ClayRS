from unittest import TestCase

from clayrs.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from textblob import TextBlob
import numpy as np


class TestTextBlobSentimentAnalysis(TestCase):
    def test_fit(self):
        text_reviews = ['good item', 'it was awful', 'pretty good', 'extraordinary', 'too much expensive']

        result = [TextBlobSentimentAnalysis().fit(text) for text in text_reviews]
        expected = [TextBlob(field_data).sentiment.polarity for field_data in text_reviews]

        self.assertEqual(expected, result)

        result_rounded = [TextBlobSentimentAnalysis(decimal_rounding=4).fit(text) for text in text_reviews]
        expected_rounded = [np.round(TextBlob(field_data).sentiment.polarity, 4) for field_data in text_reviews]

        self.assertEqual(expected_rounded, result_rounded)

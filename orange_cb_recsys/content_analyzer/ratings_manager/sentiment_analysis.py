from abc import abstractmethod
from typing import List

from textblob import TextBlob
import numpy as np

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import SentimentAnalysis


class TextBlobSentimentAnalysis(SentimentAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.
    """

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    @abstractmethod
    def __repr__(self):
        return f'TextBlobSentimentAnalysis'

    def fit(self, score_column_data: List[str]) -> List[float]:
        """
        This method calculates the sentiment analysis score on textual reviews

        Returns:
            sentiment_data: a list of sentiment analysis score
        """
        if self.decimal_rounding:
            polarity_scores = [np.round(TextBlob(field_data).sentiment.polarity, self.decimal_rounding)
                               for field_data in score_column_data]
        else:
            polarity_scores = [TextBlob(field_data).sentiment.polarity for field_data in score_column_data]

        return polarity_scores

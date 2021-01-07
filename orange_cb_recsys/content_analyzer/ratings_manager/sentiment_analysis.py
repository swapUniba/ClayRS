from textblob import TextBlob

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import SentimentAnalysis


class TextBlobSentimentAnalysis(SentimentAnalysis):
    """
    Interface for the textblob library that does sentimental analysis on text.
    """

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return "< TextBlobSentimentalAnalysis >"

    def fit(self, field_data: str) -> float:
        """
        This method calculates the sentiment analysis score on textual reviews

        Returns:
            sentiment_data: a list of sentiment analysis score
        """

        return TextBlob(field_data).sentiment.polarity

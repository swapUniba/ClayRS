from textblob import TextBlob

from clayrs.content_analyzer.ratings_manager.score_processor import SentimentAnalysis


class TextBlobSentimentAnalysis(SentimentAnalysis):
    """
    Class that compute sentiment polarity on a textual field using TextBlob library.

    The given score will be in the $[-1.0, 1.0]$ range
    """
    def __init__(self, decimal_rounding: int = None):
        super().__init__(decimal_rounding)

    def __str__(self):
        return "TextBlobSentimentalAnalysis"

    def __repr__(self):
        return f'TextBlobSentimentAnalysis'

    def fit(self, score_data: str) -> float:
        """
        This method calculates the sentiment polarity score on textual reviews

        Args:
            score_data: text for which sentiment polarity must be computed and considered as score

        Returns:
            The sentiment polarity of the textual data in range $[-1.0, 1.0]$
        """
        polarity_score = TextBlob(score_data).sentiment.polarity

        if self.decimal_rounding:
            polarity_score = round(polarity_score, self.decimal_rounding)

        return polarity_score

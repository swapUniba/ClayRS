from abc import ABC, abstractmethod
from typing import List
import numpy as np


class RatingProcessor(ABC):
    """
    Abstract class to process a rating with the personalized fit method
    that returns a score in the range [-1.0,1.0]
    """
    def __init__(self, decimal_rounding: int = None):
        self.__decimal_rounding = decimal_rounding

    @property
    def decimal_rounding(self):
        return self.__decimal_rounding

    @abstractmethod
    def fit(self, score_column_data: List[object]):
        raise NotImplementedError


class SentimentAnalysis(RatingProcessor):
    """
    Abstract Class that generalizes the sentiment analysis technique
    """

    @abstractmethod
    def fit(self, score_column_data: List[str]):
        raise NotImplementedError


class NumberNormalizer(RatingProcessor):
    """
    Class that normalizes the ratings to a numeric scale in the range [-1.0,1.0]

    Args:
        min_ (float): minimum value of the original scale
        max_ (float): maximum value of the original scale
    """
    def __str__(self):
        return "NumberNormalizer"

    def __repr__(self):
        return "< NumberNormalizer >"

    def fit(self, score_column_data: List[float]):
        """

        Args:
            field_data: rating field that will be
                normalized

        Returns:
            (float): field_data normalized in the interval [-1, 1]
        """
        def convert_into_range(value: float, old_min: float, old_max: float, new_min: int = -1, new_max: int = 1):
            new_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            if self.decimal_rounding:
                new_value = np.round(new_value, self.decimal_rounding)

            return new_value

        score_column_data = [float(score) for score in score_column_data]

        old_min = min(score_column_data)
        old_max = max(score_column_data)

        return [convert_into_range(value, old_min, old_max) for value in score_column_data]

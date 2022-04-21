from abc import ABC, abstractmethod
from typing import Tuple
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
    def fit(self, score_data: object):
        raise NotImplementedError


class SentimentAnalysis(RatingProcessor):
    """
    Abstract Class that generalizes the sentiment analysis technique
    """

    @abstractmethod
    def fit(self, score_data: str):
        raise NotImplementedError


class NumberNormalizer(RatingProcessor):
    """
    Class that normalizes the ratings to a numeric scale in the range [-1.0,1.0]

    Args:
        min_ (float): minimum value of the original scale
        max_ (float): maximum value of the original scale
    """
    def __init__(self, scale: Tuple[float, float], decimal_rounding: int = None):
        super().__init__(decimal_rounding)

        if len(scale) != 2:
            raise ValueError("The voting scale should be a tuple containing exactly two values,"
                             "the minimum of the scale and the maximum!")

        self._old_min = scale[0]
        self._old_max = scale[1]

    def __str__(self):
        return "NumberNormalizer"

    def __repr__(self):
        return "< NumberNormalizer >"

    def fit(self, score_data: float):
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

        return convert_into_range(float(score_data), self._old_min, self._old_max)

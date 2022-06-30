from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ScoreProcessor(ABC):
    """
    Abstract class to process a rating with the personalized fit method
    """
    def __init__(self, decimal_rounding: int = None):
        self.__decimal_rounding = decimal_rounding

    @property
    def decimal_rounding(self):
        return self.__decimal_rounding

    @abstractmethod
    def fit(self, score_data: object):
        raise NotImplementedError

    def __repr__(self):
        return f'ScoreProcessor(decimal rounding={self.__decimal_rounding})'


class SentimentAnalysis(ScoreProcessor):
    """
    Abstract Class that generalizes the sentiment analysis technique
    """

    @abstractmethod
    def fit(self, score_data: str):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        return f'SentimentAnalysis(decimal rounding={self.decimal_rounding})'


class NumberNormalizer(ScoreProcessor):
    """
    Class that normalizes numeric scores to a scale in the range $[-1.0, 1.0]$

    Args:
        scale: Tuple where the first value is the minimum of the actual scale, second value is the maximum of the
            actual scale (e.g. `(1, 5)` represents an actual scale of scores from 1 (included) to 5 (included))
        decimal_rounding: If set, the normalized score will be rounded to the chosen decimal digit
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
        return f'NumberNormalizer(scale=({self._old_min}, {self._old_max}), decimal rounding={self.decimal_rounding})'

    def fit(self, score_data: float) -> float:
        """
        Method which will normalize the given score

        Args:
            score_data: score that will be normalized

        Returns:
            score normalized in the interval $[-1, 1]$
        """
        def convert_into_range(value: float, old_min: float, old_max: float, new_min: int = -1, new_max: int = 1):
            new_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            if self.decimal_rounding:
                new_value = np.round(new_value, self.decimal_rounding)

            return new_value

        return convert_into_range(float(score_data), self._old_min, self._old_max)

from abc import ABC, abstractmethod

import pandas as pd


class Metric(ABC):
    """
    Abstract class that generalize metric concept;
    """

    @abstractmethod
    def perform(self, predictions: pd.DataFrame, truth: pd.DataFrame):
        """
        Method that execute the metric computation

        Args:
              truth (pd.DataFrame): dataframe with known ratings,
                  it is used as ground truth in metric computation
                  predictions (pd.DataFrame): dataframe with predicted items and
                  associated scores
        """
        raise NotImplementedError


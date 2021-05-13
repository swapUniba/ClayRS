import abc
from abc import ABC
from typing import List

import pandas as pd


class Algorithm(ABC):
    """
    Abstract class for an Algorithm
    """

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Method to call right after the instantiation of the algorithm.

        Must be defined for every type of algorithm implemented, its task is to pass important parameters
        to the algorithm, such as the graph for the graph-based algorithms, or the ratings, items path and
        user path for the content-based algorithms
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_predict(self, user_id: str, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method which predicts the ratings of a user for the unrated items.

        If the 'filter_list' parameter is passed then the rating is predicted for those items,
        otherwise all unrated items will be predicted.

        Args:
            user_id (str): user for which the predictions will be calculated
            filter_list (list): list of items that will be predicted. If None,
                all items will be predicted
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_rank(self, user_id: str, recs_number: int = None, filter_list: List[str] = None) -> pd.DataFrame:
        """
        Method which predicts the ratings of a user for the unrated items and ranks them.

        If recs_number parameter is specified, only the top-n items are shown, where n is the recs_number.

        If the 'filter_list' parameter is passed then the rating is predicted for those items,
        otherwise all unrated items will be predicted and ranked.
        Args:
            user_id (str): user for which the rank will be calculated
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of items that will be ranked. If None,
                all items will be ranked
        """
        raise NotImplementedError

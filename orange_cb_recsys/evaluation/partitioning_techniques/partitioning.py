import abc
from abc import ABC
from typing import Tuple

from sklearn.model_selection import KFold, train_test_split
import pandas as pd

from orange_cb_recsys.evaluation.exceptions import PartitionError


class Partitioning(ABC):
    """
    Abstract Class for partitioning technique
    """

    def __init__(self):
        self._dataframe: pd.DataFrame = None

    @property
    def dataframe(self):
        return self._dataframe

    @abc.abstractmethod
    def __iter__(self) -> Tuple[pd.DataFrame]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_dataframe(self, dataframe: pd.DataFrame):
        raise NotImplementedError


class KFoldPartitioning(Partitioning):
    """
    Class that perform K-Fold partitioning

    Args:
        n_splits (int): number of splits
        random_state (int): random state
    """

    def __init__(self, n_splits: int = 2, random_state: int = 42):
        self.__n_splits = n_splits
        self.__random_state = random_state

        super().__init__()

    def set_dataframe(self, dataframe: pd.DataFrame):
        if len(dataframe) < self.__n_splits:
            raise PartitionError("Number of splits larger than number of frame rows")
        else:
            self._dataframe = dataframe

    def __iter__(self) -> Tuple[pd.DataFrame]:
        kf = KFold(n_splits=self.__n_splits, shuffle=True, random_state=self.__random_state)
        split_result = kf.split(self.dataframe)

        # iloc because split_result are list of ints
        for train_index, test_index in split_result:
            yield self.dataframe.iloc[train_index], self.dataframe.iloc[test_index]


class HoldOutPartitioning(Partitioning):
    """
    Class that perform Hold-Out partitioning

    Args:
        train_set_size (float): percentage of how much big in percentage the train set must be
            EXAMPLE: train_set_size = 0.8, train_set_size = 0.65, train_set_size = 0.2
        random_state (int): random state
    """

    def __init__(self, train_set_size: float = 0.8, random_state: int = 42):
        self._check_percentage(train_set_size)
        self.__train_set_size = train_set_size
        self.__test_set_size = (1 - train_set_size)
        self.__random_state = random_state

        super().__init__()

    @staticmethod
    def _check_percentage(percentage: float):
        if (percentage <= 0) or (percentage >= 1):
            raise ValueError("The train set size must be a float in the (0, 1) interval")

    def set_dataframe(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe

    def __iter__(self) -> Tuple[pd.DataFrame]:
        train_index, test_index = train_test_split(self.dataframe.index,
                                                   train_size=self.__train_set_size,
                                                   test_size=self.__test_set_size,
                                                   shuffle=True,
                                                   random_state=self.__random_state)

        # loc because split_result are Indexes so we must search by labels
        yield self.dataframe.loc[train_index], self.dataframe.loc[test_index]

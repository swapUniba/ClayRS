from abc import ABC
from sklearn.model_selection import KFold
import pandas as pd


class Partitioning(ABC):
    """
    Abstract Class for partitioning technique
    """
    def __init__(self):
        self.__dataframe: pd.DataFrame = None

    def __iter__(self):
        raise NotImplementedError

    @property
    def dataframe(self):
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame):
        self.__dataframe = dataframe


class KFoldPartitioning(Partitioning):
    """
    Class that perform K-Fold partitioning

    Args:
        n_splits (int): number of splits
        random_state (int): random state
    """
    def __init__(self, n_splits: int = 2, random_state: int = 2):
        super().__init__()
        self.__n_splits = n_splits
        self.__random_state = random_state

    def set_dataframe(self, dataframe: pd.DataFrame):
        if len(dataframe) < self.__n_splits:
            raise ValueError("Number of splits larger than number of frame rows")
        self.dataframe = dataframe

    def __iter__(self):
        kf = KFold(n_splits=self.__n_splits, shuffle=True, random_state=self.__random_state)
        split_result = kf.split(self.dataframe)

        for result in split_result:
            yield result

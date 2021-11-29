import pandas as pd
from typing import Set

import abc
from abc import ABC
from typing import Tuple

from sklearn.model_selection import KFold, train_test_split

#from orange_cb_recsys.recsys.partitioning import PartitionError
from orange_cb_recsys.utils.const import eval_logger, progbar


class Split:
    """
    Class container for two pandas DataFrame

    It may represent a split containing 'train set' and 'test set', or a split containing a ground truth and predictions
    for it, etc.

    Once instantiated, one can access the two dataframes in different ways:

    | > sp = Split()
    | > # Various ways of accessing the FIRST DataFrame
    | > sp.train
    | > sp.pred
    | > sp.first
    | >
    | > # Various ways of accessing the SECOND DataFrame
    | > sp.test
    | > sp.truth
    | > sp.second

    Args:
        first_set (pd.DatFrame): the first DataFrame to contain. If not specified, an empty DataFrame with 'from_id',
            'to_id', and 'score' column will be instantiated
        second_set (pd.DataFrame): the second DataFrame to contain. If not specified, an empty DataFrame with 'from_id',
            'to_id' and 'score' column will be instantiated
    """

    def __init__(self,
                 first_set=pd.DataFrame({'from_id': [], 'to_id': [], 'score': []}),
                 second_set=pd.DataFrame({'from_id': [], 'to_id': [], 'score': []})):

        self.__dict__['first'] = first_set
        self.__dict__['second'] = second_set

        self.__dict__['_valid_first_name'] = ['train', 'pred', 'first']
        self.__dict__['_valid_second_name'] = ['test', 'truth', 'second']

    def __getattr__(self, name):
        if name in self._valid_first_name:
            return self.first
        elif name in self._valid_second_name:
            return self.second

    def __setattr__(self, name, value):
        if name in self._valid_first_name:
            super().__setattr__('first', value)
        elif name in self._valid_second_name:
            super().__setattr__('second', value)


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

    @abc.abstractmethod
    def __str__(self):
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

    def __str__(self):
        return "KFoldPartitioningTechnique"


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

    def __str__(self):
        return "HoldOutPartitioningTechnique"


class PartitionModule:
    """
    Module of the Evaluation pipeline which has the task of splitting the original interactions in 'train set' and 'test
    set'.

    Different kinds of partitioning technique may be used, check the correspondent documentation for more

    Args:
        partition_technique (Partitioning): The technique that will be used to split original interactions in
            'train set' and 'test set'.
    """

    def __init__(self, partition_technique: Partitioning):
        self._partition_technique = partition_technique

    def _split_single(self, user_ratings: pd.DataFrame):
        """
        Private method that splits the ratings of a single user into 'train set' and 'test set'

        Args:
            user_ratings (pd.DataFrame): DataFrame containing the ratings of a single user that will be splitted into
                'train set' and 'test set'
        """

        self._partition_technique.set_dataframe(user_ratings)  # May raise exception

        user_splits = [Split(train_set, test_set) for train_set, test_set in self._partition_technique]
        return user_splits

    def split_all(self, ratings: pd.DataFrame, user_id_list: Set[str]):
        """
        Method that effectively splits the 'ratings' parameter into 'train set' and 'test set'.
        It must be specified a 'user_id_list' parameter so that the method will do the splitting only for the users
        specified inside the list.

        Args:
            ratings (pd.DataFrame): The DataFrame which contains the interactions of the users that must be splitted
                into 'train set' and 'test set'
            user_id_list (Set[str]): The set of users for which splitting will be done
        """

        split_list = []

        eval_logger.info("Performing {} on ratings of every user".format(str(self._partition_technique)))
        for user_id in progbar(user_id_list, prefix="Current user - {}:", substitute_with_current=True):
            user_ratings = ratings[ratings['from_id'] == user_id]
            try:
                user_splits_list = self._split_single(user_ratings)
            except PartitionError as e:
                eval_logger.warning(str(e) + "\nThe user {} will be skipped".format(user_id))
                continue

            if len(split_list) != 0:
                for user_split, total_split in zip(user_splits_list, split_list):
                    total_split.train = pd.concat([total_split.train, user_split.train])
                    total_split.test = pd.concat([total_split.test, user_split.test])
            else:
                for user_split in user_splits_list:
                    split_list.append(user_split)  # Only executed once

        return split_list



class PartitionError(Exception):
    """
    Exception to raise when ratings of a user can't be split, e.g. (n_splits > n_user_ratings)
    """
    pass

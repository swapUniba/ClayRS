from collections import defaultdict

import pandas as pd
from typing import Set

import abc
from abc import ABC

from sklearn.model_selection import KFold, train_test_split

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
    def __init__(self, skip_user_error: bool = True):
        self.__skip_user_error = skip_user_error

    @property
    def skip_user_error(self):
        return self.__skip_user_error

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def split_single(self, user_ratings: pd.DataFrame):
        raise NotImplementedError

    def split_all(self, ratings: pd.DataFrame, user_id_list: Set[str] = None):
        """
        Method that effectively splits the 'ratings' parameter into 'train set' and 'test set'.
        It must be specified a 'user_id_list' parameter so that the method will do the splitting only for the users
        specified inside the list.

        Args:
            ratings (pd.DataFrame): The DataFrame which contains the interactions of the users that must be splitted
                into 'train set' and 'test set'
            user_id_list (Set[str]): The set of users for which splitting will be done
        """

        if user_id_list is None:
            user_id_list = set(ratings['from_id'])

        # {0: {'train': [train1_u1, train1_u2], 'test': [test1_u1, test1_u2]},
        #  1: {'train': [train2_u1, train2_u2], 'test': [test2_u1, test2_u2]}}
        train_test_dict = defaultdict(lambda: defaultdict(list))

        eval_logger.info("Performing {}".format(str(self)))
        for user_id in progbar(user_id_list, prefix="Current user - {}:", substitute_with_current=True):
            user_ratings = ratings[ratings['from_id'] == user_id]
            try:
                user_train_list, user_test_list = self.split_single(user_ratings)
                for i, (single_train, single_test) in enumerate(zip(user_train_list, user_test_list)):
                    train_test_dict[i]['train'].append(single_train)
                    train_test_dict[i]['test'].append(single_test)

            except ValueError as e:
                if self.skip_user_error:
                    eval_logger.warning(str(e) + "\nThe user {} will be skipped".format(user_id))
                    continue
                else:
                    raise e

        train_list = [pd.concat(train_test_dict[split]['train']) for split in train_test_dict]
        test_list = [pd.concat(train_test_dict[split]['test']) for split in train_test_dict]

        return train_list, test_list


class KFoldPartitioning(Partitioning):
    """
    Class that perform K-Fold partitioning

    Args:
        n_splits (int): Number of splits. Must be at least 2
        random_state (int): random state
    """

    def __init__(self, n_splits: int = 2, shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):
        self.__kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        super(KFoldPartitioning, self).__init__(skip_user_error)

    def split_single(self, user_ratings: pd.DataFrame):

        index_dataframe = user_ratings.index.to_numpy()

        split_result = self.__kf.split(index_dataframe)

        user_train_list = []
        user_test_list = []

        for train_index, test_index in split_result:

            # loc since we are accessing by position
            user_train_list.append(user_ratings.iloc[train_index])
            user_test_list.append(user_ratings.iloc[test_index])

        return user_train_list, user_test_list

    def __str__(self):
        return "KFoldPartitioningTechnique"


class HoldOutPartitioning(Partitioning):
    """
    Class that perform Hold-Out partitioning

    Args:
        train_set_size (float): percentage of how much big in percentage the train set of each user must be
            EXAMPLE: train_set_size = 0.8, train_set_size = 0.65, train_set_size = 0.2
        random_state (int): random state
    """

    def __init__(self, train_set_size: float = 0.8, shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):
        self._check_percentage(train_set_size)
        self.__train_set_size = train_set_size
        self.__test_set_size = (1 - train_set_size)
        self.__random_state = random_state
        self.__shuffle = shuffle

        super().__init__(skip_user_error)

    @staticmethod
    def _check_percentage(percentage: float):
        if (percentage <= 0) or (percentage >= 1):
            raise ValueError("The train set size must be a float in the (0, 1) interval")

    def split_single(self, user_ratings: pd.DataFrame):
        index_to_split = user_ratings.index.to_numpy()

        train_index, test_index = train_test_split(index_to_split,
                                                   train_size=self.__train_set_size,
                                                   test_size=self.__test_set_size,
                                                   shuffle=self.__shuffle,
                                                   random_state=self.__random_state)

        user_train_list = [user_ratings.loc[train_index]]
        user_test_list = [user_ratings.loc[test_index]]

        # loc since we are accessing by label
        return user_train_list, user_test_list

    def __str__(self):
        return "HoldOutPartitioningTechnique"

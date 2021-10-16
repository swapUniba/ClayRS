from typing import Set

import pandas as pd

from orange_cb_recsys.evaluation.exceptions import PartitionError
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import Partitioning
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

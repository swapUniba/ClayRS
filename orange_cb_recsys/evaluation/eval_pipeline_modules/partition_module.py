from typing import Set

import pandas as pd

from orange_cb_recsys.evaluation.exceptions import PartitionError
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import Partitioning
from orange_cb_recsys.utils.const import logger


class Split:
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

    def __init__(self, partition_technique: Partitioning):
        self._partition_technique = partition_technique

    def _split_single(self, user_ratings: pd.DataFrame):

        self._partition_technique.set_dataframe(user_ratings)  # May raise exception

        user_splits = [Split(train_set, test_set) for train_set, test_set in self._partition_technique]
        return user_splits

    def split_all(self, ratings: pd.DataFrame, user_id_list: Set[str]):

        split_list = []

        for user_id in user_id_list:
            user_ratings = ratings[ratings['from_id'] == user_id]
            try:
                user_splits_list = self._split_single(user_ratings)
            except PartitionError as e:
                logger.warning(str(e) + "\nThe user {} will be skipped".format(user_id))
                continue

            if len(split_list) != 0:
                for user_split, total_split in zip(user_splits_list, split_list):
                    total_split.train = pd.concat([total_split.train, user_split.train])
                    total_split.test = pd.concat([total_split.test, user_split.test])
            else:
                for user_split in user_splits_list:
                    split_list.append(user_split)  # Only executed once

        return split_list

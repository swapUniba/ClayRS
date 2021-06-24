import abc
from abc import ABC
from typing import List, Set
import pandas as pd

from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split


class Methodology(ABC):

    def get_item_to_predict(self, split_list: List[Split]) -> List[pd.DataFrame]:

        items_to_predict = []
        for split in split_list:
            user_list = set(split.truth.from_id)

            single_split_items = {'from_id': [], 'to_id': []}

            for user in user_list:
                single_user_to_id = self._get_single_user_to_id(user, split)

                single_user_from_id = [user for i in range(len(single_user_to_id))]

                single_split_items['from_id'].extend(single_user_from_id)

                single_split_items['to_id'].extend(single_user_to_id)

            items_to_predict.append(pd.DataFrame(single_split_items))

        return items_to_predict

    @abc.abstractmethod
    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        raise NotImplementedError


class TestRatingsMethodology(Methodology):
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    @property
    def threshold(self):
        return self.__threshold

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        if self.__threshold is not None:
            single_user_to_id = set(split.test.query('(from_id == @user) '
                                                     'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.test.query('from_id == @user').to_id)

        return single_user_to_id


class TestItemsMethodology(Methodology):
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    @property
    def threshold(self):
        return self.__threshold

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        # variable used by pandas query
        user_ratings_train = split.train.query('from_id == @user')

        if self.__threshold is not None:
            single_user_to_id = set(split.test.query('(to_id not in @user_ratings_train.to_id) '
                                                     'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.test.query('to_id not in @user_ratings_train.to_id').to_id)

        return single_user_to_id


class TrainingItemsMethodology(Methodology):
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    @property
    def threshold(self):
        return self.__threshold

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        # variable used by pandas query
        user_ratings_train = split.train.query('from_id == @user')

        if self.__threshold is not None:
            single_user_to_id = set(split.train.query('(to_id not in @user_ratings_train.to_id) '
                                                      'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.train.query('to_id not in @user_ratings_train.to_id').to_id)

        return single_user_to_id


class AllItemsMethodology(Methodology):

    def __init__(self, items_list: Set[str]):
        self.__items_list = items_list

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:

        user_ratings_train = split.train.query('from_id == @user')

        single_user_to_id = set([item for item in self.__items_list if item not in set(user_ratings_train.to_id)])

        return single_user_to_id

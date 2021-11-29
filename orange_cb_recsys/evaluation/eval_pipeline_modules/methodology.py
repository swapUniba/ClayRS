import abc
from abc import ABC
from typing import List, Set
import pandas as pd

from orange_cb_recsys.recsys.partitioning import Split
from orange_cb_recsys.utils.const import eval_logger


class Methodology(ABC):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    The methodologies here implemented follow the 'Precision-Oriented Evaluation of Recommender Systems: An Algorithmic
    Comparison' paper, check it for more
    """

    def get_item_to_predict(self, split_list: List[Split]) -> List[pd.DataFrame]:
        """
        Method which effectively calculates which items must be used in order to generate a recommendation list

        It takes in input all splits containing 'train set' and 'test set' and returns a list of DataFrame, one for
        every split. A single DataFrame contains, for every user inside the train set, all items which must be
        recommended based on the methodology chosen.

        Args:
            split_list (List[Split]): List of split where every split contains a 'train set' and a 'test set'.

        Returns:
            A list of DataFrame, one for every split. A single DataFrame contains all items which must be
            recommended to every user based on the methodology chosen.
        """

        items_to_predict = []
        for counter, split in enumerate(split_list, start=1):
            eval_logger.info("Getting items to predict with {} for split {}".format(str(self), counter))

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
        """
        Abstract method in which must be specified how to calculate which items must be part of the recommendation list
        of a single user
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class TestRatingsMethodology(Methodology):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    With TestRatingsMethodology, given a user U, items to recommend for U are simply those items that appear in its
    'test set'
    \n

    If the 'only_greater_eq' parameter is specified, then only items with rating score >= only_greater_eq will be
    returned

    Args:
        only_greater_eq (float): float which acts as a filter, if specified only items with
            rating score >= only_greater_eq will be returned
    """
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    def __str__(self):
        return "TestRatingsMethodology"

    @property
    def threshold(self):
        return self.__threshold

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        """
        Private method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TestRatings Methodology, only items that appear in the 'test set' of the user will be returned

        Args:
            user (str): User of which we want to calculate items that must appear in its recommendation list
            split (Split): Split containing 'train set' and 'test set'
        """
        if self.__threshold is not None:
            single_user_to_id = set(split.test.query('(from_id == @user) '
                                                     'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.test.query('from_id == @user').to_id)

        return single_user_to_id


class TestItemsMethodology(Methodology):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    With TestItemsMethodology, given a user U, items to recommend for U are all items that appear in the 'test set' of
    every user excluding those items that appear in the 'train set' of U
    \n
    If the 'only_greater_eq' parameter is specified, then only items with rating score >= only_greater_eq will be
    returned

    Args:
        only_greater_eq (float): float which acts as a filter, if specified only items with
            rating score >= only_greater_eq will be returned
    """
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    @property
    def threshold(self):
        return self.__threshold

    def __str__(self):
        return "TestItemsMethodology"

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        """
        Private method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TestItems Methodology, all items that appear in the 'test set' of every user will be returned,
        except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user (str): User of which we want to calculate items that must appear in its recommendation list
            split (Split): Split containing 'train set' and 'test set'
        """
        # variable used by pandas query
        user_ratings_train = split.train.query('from_id == @user')

        if self.__threshold is not None:
            single_user_to_id = set(split.test.query('(to_id not in @user_ratings_train.to_id) '
                                                     'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.test.query('to_id not in @user_ratings_train.to_id').to_id)

        return single_user_to_id


class TrainingItemsMethodology(Methodology):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    With TrainingItemsMethodology, given a user U, items to recommend for U are all items that appear in the 'train set'
    of every user excluding those items that appear in the 'train set' of U
    \n
    If the 'only_greater_eq' parameter is specified, then only items with rating score >= only_greater_eq will be
    returned

    Args:
        only_greater_eq (float): float which acts as a filter, if specified only items with
            rating score >= only_greater_eq will be returned
    """
    def __init__(self, only_greater_eq: float = None):
        self.__threshold = only_greater_eq

    @property
    def threshold(self):
        return self.__threshold

    def __str__(self):
        return "TrainingItemsMethodology"

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        """
        Private method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TrainingItems Methodology, all items that appear in the 'test set' of every user will be returned,
        except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user (str): User of which we want to calculate items that must appear in its recommendation list
            split (Split): Split containing 'train set' and 'test set'
        """
        # variable used by pandas query
        user_ratings_train = split.train.query('from_id == @user')

        if self.__threshold is not None:
            single_user_to_id = set(split.train.query('(to_id not in @user_ratings_train.to_id) '
                                                      'and (score >= @self.threshold)').to_id)
        else:
            single_user_to_id = set(split.train.query('to_id not in @user_ratings_train.to_id').to_id)

        return single_user_to_id


class AllItemsMethodology(Methodology):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    With AllItemsMethodology, given a user U, items to recommend for U are all items that appear in 'items_list'
    parameter excluding those items that appear in the 'train set' of U
    \n
    If the 'only_greater_eq' parameter is specified, then only items with rating score >= only_greater_eq will be
    returned

    Args:
        items_list (Set[str]): Items set that must appear in the recommendation list of every user
    """

    def __init__(self, items_list: Set[str]):
        self.__items_list = items_list

    def __str__(self):
        return "AllItemsMethodology"

    def _get_single_user_to_id(self, user: str, split: Split) -> Set:
        """
        Private method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the AllItems Methodology, all items that appear in the 'items_list' parameter of the constructor
        will be returned, except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user (str): User of which we want to calculate items that must appear in its recommendation list
            split (Split): Split containing 'train set' and 'test set'
        """

        user_ratings_train = split.train.query('from_id == @user')

        single_user_to_id = set([item for item in self.__items_list if item not in set(user_ratings_train.to_id)])

        return single_user_to_id

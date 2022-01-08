import abc
from abc import ABC
from typing import Set, Union, Dict
import pandas as pd

from orange_cb_recsys.utils.const import logger


class Methodology(ABC):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    The methodologies here implemented follow the 'Precision-Oriented Evaluation of Recommender Systems: An Algorithmic
    Comparison' paper, check it for more
    """

    def filter_all(self, train_set: pd.DataFrame, test_set: pd.DataFrame,
                   result_as_dict: bool = False) -> Union[pd.DataFrame, Dict]:
        """
        Method which effectively calculates which items must be used in order to generate a recommendation list

        It takes in input a 'train set' and a 'test set' and returns a single DataFrame which contains,
        for every user inside the train set, all items which must be recommended based on the methodology chosen.

        Args:
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
            result_as_dict (bool): If True the output of the method will be a dict containing users as a key and
                list of item that must be predicted as a value.
                EXAMPLE:
                    {'u1': ['i1', 'i2', 'i3'], 'u2': ['i1', 'i4'], ...}
        Returns:
            A DataFrame which contains all items which must be recommended to every user based on the methodology
            chosen.
        """
        logger.info("Filtering items based on methodology chosen...")

        user_list = set(test_set.from_id)

        filtered_frames_to_concat = [self.filter_single(user, train_set, test_set)
                                     for user in user_list]

        filter_frame = pd.concat(filtered_frames_to_concat)[['from_id', 'to_id']]
        if result_as_dict:
            filter_frame = dict(filter_frame.groupby('from_id')['to_id'].apply(list))

        return filter_frame

    @abc.abstractmethod
    def filter_single(self, user_id: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
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

    def filter_single(self, user_id: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestRatings Methodology, only items that appear in the 'test set' of the user will be returned.

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas Dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        user_test = test_set[test_set['from_id'] == user_id]

        if self.__threshold is not None:
            filtered_user_test = user_test[user_test['score'] >= self.threshold]
        else:
            # TestRatings just returns the test set of the user
            filtered_user_test = user_test

        return filtered_user_test


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

    def filter_single(self, user_id: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestItems Methodology, all items that appear in the 'test set' of every user will be returned,
        except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """

        already_seen_items = set(train_set[train_set['from_id'] == user_id].to_id)

        if self.__threshold is not None:
            filtered_items = set(test_set[(~test_set['to_id'].isin(already_seen_items))
                                          &
                                          (test_set['score'] >= self.threshold)].to_id)
        else:
            filtered_items = set(test_set[~test_set['to_id'].isin(already_seen_items)].to_id)

        filtered_user_test = pd.DataFrame({'from_id': user_id, 'to_id': list(filtered_items)})

        return filtered_user_test


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

    def filter_single(self, user_id: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TrainingItems Methodology, all items that appear in the 'train set' of every user will be
        returned, except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        already_seen_items = set(train_set[train_set['from_id'] == user_id].to_id)

        if self.__threshold is not None:
            filtered_items = set(train_set[(~train_set['to_id'].isin(already_seen_items))
                                           &
                                           (train_set['score'] >= self.threshold)].to_id)
        else:
            filtered_items = set(train_set[~train_set['to_id'].isin(already_seen_items)].to_id)

        filtered_user_test = pd.DataFrame({'from_id': user_id, 'to_id': list(filtered_items)})

        return filtered_user_test


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

    def filter_single(self, user_id: str, train_set: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the AllItems Methodology, all items that appear in the 'items_list' parameter of the constructor
        will be returned, except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        already_seen_items = set(train_set[train_set['from_id'] == user_id].to_id)

        filtered_items = set([item for item in self.__items_list if item not in already_seen_items])

        filtered_user_test = pd.DataFrame({'from_id': user_id, 'to_id': list(filtered_items)})

        return filtered_user_test

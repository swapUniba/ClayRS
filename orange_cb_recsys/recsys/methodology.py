import abc
import itertools
from abc import ABC
from typing import Set, Union, Dict
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from orange_cb_recsys.content_analyzer import Ratings
from orange_cb_recsys.utils.const import logger, get_pbar


class Methodology(ABC):
    """
    Module of the Evaluation pipeline which, given a 'train set' and a 'test set', has the task to calculate
    which items must be used in order to generate a recommendation list

    The methodologies here implemented follow the 'Precision-Oriented Evaluation of Recommender Systems: An Algorithmic
    Comparison' paper, check it for more
    """
    def __init__(self, only_greater_eq: float = None):
        self._threshold = only_greater_eq

    def _filter_only_greater_eq(self, split_set: Ratings):
        interaction_list_greater_eq = [interaction for interaction in split_set if interaction.score >= self._threshold]

        return Ratings.from_list(interaction_list_greater_eq)

    def filter_all(self, train_set: Ratings, test_set: Ratings,
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
        user_list = set(test_set.user_id_column)

        with logging_redirect_tqdm():
            pbar = get_pbar(user_list)
            pbar.set_description(f"Filtering items based on {str(self)}")

            filtered = {user_id: self.filter_single(user_id, train_set, test_set)
                        for user_id in pbar}

        if not result_as_dict:
            filtered = pd.DataFrame([(user_id, item_to_predict)
                                     for user_id, all_items_to_pred in zip(filtered.keys(), filtered.values())
                                     for item_to_predict in all_items_to_pred],
                                    columns=['user_id', 'item_id'])

        return filtered

    @abc.abstractmethod
    def filter_single(self, user_id: str, train_set: Ratings, test_set: Ratings) -> pd.DataFrame:
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
        super(TestRatingsMethodology, self).__init__(only_greater_eq)

    def __str__(self):
        return "TestRatingsMethodology"

    def filter_single(self, user_id: str, train_set: Ratings, test_set: Ratings) -> Set:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestRatings Methodology, only items that appear in the 'test set' of the user will be returned.

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas Dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        user_test = test_set.get_user_interactions(user_id)

        if self._threshold is not None:
            filtered_items = set([interaction.item_id
                                  for interaction in user_test if interaction.score >= self._threshold])
        else:
            # TestRatings just returns the test set of the user
            filtered_items = set([interaction.item_id for interaction in user_test])

        return filtered_items


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
        super(TestItemsMethodology, self).__init__(only_greater_eq)

        self._filtered_test_set: Union[Ratings, None] = None

    def __str__(self):
        return "TestItemsMethodology"

    def filter_single(self, user_id: str, train_set: Ratings, test_set: Ratings) -> Set:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestItems Methodology, all items that appear in the 'test set' of every user will be returned,
        except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        already_seen_items = set([interaction.item_id for interaction in train_set.get_user_interactions(user_id)])

        filtered_test_set = test_set
        if self._threshold is not None:
            if self._filtered_test_set is None:
                self._filtered_test_set = self._filter_only_greater_eq(test_set)

            filtered_test_set = self._filtered_test_set

        filtered_items = set(filtered_test_set.item_id_column) - already_seen_items

        return filtered_items


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
        super(TrainingItemsMethodology, self).__init__(only_greater_eq)

        self._filtered_train_set: Union[Ratings, None] = None

    def __str__(self):
        return "TrainingItemsMethodology"

    def filter_single(self, user_id: str, train_set: Ratings, test_set: Ratings) -> Set:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TrainingItems Methodology, all items that appear in the 'train set' of every user will be
        returned, except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        already_seen_items = set([interaction.item_id for interaction in train_set.get_user_interactions(user_id)])

        filtered_train_set = train_set
        if self._threshold is not None:
            if self._filtered_train_set is None:
                self._filtered_train_set = self._filter_only_greater_eq(train_set)

            filtered_train_set = self._filtered_train_set

        filtered_items = set(filtered_train_set.item_id_column) - already_seen_items

        return filtered_items


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
        self._items_list = items_list
        super(AllItemsMethodology, self).__init__(None)

    def __str__(self):
        return "AllItemsMethodology"

    def filter_single(self, user_id: str, train_set: Ratings, test_set: Ratings) -> Set:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the AllItems Methodology, all items that appear in the 'items_list' parameter of the constructor
        will be returned, except for those that appear in the 'train set' of the user passed as parameter

        Args:
            user_id (str): User of which we want to calculate items that must appear in its recommendation list
            train_set (pd.DataFrame): Pandas dataframe which contains the train set of every user
            test_set (pd.DataFrame): Pandas dataframe which contains the test set of every user
        """
        already_seen_items = set([interaction.item_id for interaction in train_set.get_user_interactions(user_id)])

        filtered_items = set(self._items_list) - already_seen_items

        return filtered_items

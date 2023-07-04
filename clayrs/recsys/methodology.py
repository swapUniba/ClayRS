from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Set, Union, Optional, Dict, TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings

from clayrs.utils.context_managers import get_progbar


class Methodology(ABC):
    """
    Class which, given a *train set* and a *test set*, has the task to calculate which items must be used in
    order to generate a recommendation list

    The methodologies here implemented follow the 'Precision-Oriented Evaluation of Recommender Systems: An Algorithmic
    Comparison' paper
    """

    def __init__(self, only_greater_eq: float = None):

        self._threshold = only_greater_eq

        # items arr is an array with all items id mapped to their integer
        self._items_arr: Optional[np.ndarray] = None
        # query vector is the vector with same length of _items_arr used as boolean query vector
        # position in which a True appears will be taken from _items_arr, position set to False will not
        self._query_vector: Optional[np.ndarray] = None

    @abstractmethod
    def setup(self, train_set: Ratings, test_set: Ratings) -> Methodology:
        """
        Method to call before calling `filter_all()` or `filter_single()`.
        It is used to set up numpy arrays which will filter items according to the methodology chosen.

        This method has side effect, meaning that it will return a `Methodology` object which has been set up but will
        also change the `Methodology` object that has called this method

        Args:
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user

        Returns:
            The set-up Methodology object
        """
        raise NotImplementedError

    def _filter_only_greater_eq(self, split_set: Ratings):
        """
        Private method which filter the train set or the test set by considering only interactions
        which have score greater than set threshold
        """
        items_list_greater_eq = split_set.score_column >= self._threshold

        # pd.unique faster than np.unique
        return pd.unique(split_set.item_idx_column[items_list_greater_eq])

    def filter_all(self, train_set: Ratings, test_set: Ratings,
                   result_as_dict: bool = False,
                   ids_as_str: bool = True) -> Union[pd.DataFrame,
                                                     Union[Dict[str, np.ndarray], Dict[int, np.ndarray]]]:
        """
        Concrete method which calculates for all users of the *test set* which items must be used in order to
        generate a recommendation list

        It takes in input a *train set* and a *test set* and returns a single DataFrame or a python
        dictionary containing, for every user, all items which must be recommended based on the methodology chosen.

        Args:
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user
            result_as_dict: If True the output of the method will be a generator of a dictionary that contains
                users as keys and numpy arrays with items as values. If `ids_as_str` is set to True, users and items
                will be present with their string id, otherwise will be present with their mapped integer
            ids_as_str: If True, the result will contain users and items represented with their string id. Otherwise,
                will be present with their mapped integer

        Returns:
            A DataFrame or a python dictionary which contains all items which must be recommended to
                every user based on the methodology chosen.
        """
        user_list = test_set.unique_user_idx_column
        user_int2str = train_set.user_map.convert_int2str
        item_seq_int2str = train_set.item_map.convert_seq_int2str

        with get_progbar(user_list) as pbar:
            pbar.set_description(f"Filtering items based on {str(self)}")

            if ids_as_str:
                filtered = {user_int2str(user_idx): item_seq_int2str(self.filter_single(user_idx, train_set, test_set).astype(int))
                            for user_idx in pbar}
            else:
                filtered = {user_idx: self.filter_single(user_idx, train_set, test_set)
                            for user_idx in pbar}

        if not result_as_dict:

            will_be_frame = {"user_id": [], "item_id": []}
            for user_id, filter_list in filtered.items():

                will_be_frame["user_id"].append(np.full(filter_list.shape, user_id))
                will_be_frame["item_id"].append(filter_list)

            will_be_frame["user_id"] = np.hstack(will_be_frame["user_id"])
            will_be_frame["item_id"] = np.hstack(will_be_frame["item_id"])

            filtered = pd.DataFrame.from_dict(will_be_frame)

        return filtered

    @abstractmethod
    def filter_single(self, user_idx: int, train_set: Ratings, test_set: Ratings) -> np.ndarray:
        """
        Abstract method in which must be specified how to calculate which items must be part of the recommendation list
        of a single user
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class TestRatingsMethodology(Methodology):
    """
    Class which, given a *train set* and a *test set*, has the task to calculate which items must be used in
    order to generate a recommendation list

    With TestRatingsMethodology, given a user $u$, items to recommend for $u$ are simply those items that appear in its
    *test set*

    If the `only_greater_eq` parameter is set, then only items with rating score $>=$ only_greater_eq will be
    returned

    Args:
        only_greater_eq: float which acts as a filter, if specified only items with
            rating score $>=$ only_greater_eq will be returned
    """

    def __init__(self, only_greater_eq: float = None):
        super(TestRatingsMethodology, self).__init__(only_greater_eq)

    def __str__(self):
        return "TestRatingsMethodology"

    def __repr__(self):
        return f"TestRatingsMethodology(only_greater_eq={self._threshold})"

    def setup(self, train_set: Ratings, test_set: Ratings):
        return self

    def filter_single(self, user_idx: int, train_set: Ratings, test_set: Ratings) -> np.ndarray:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestRatings Methodology, only items that appear in the *test set* of the user will be returned.

        Args:
            user_idx: User idx (meaning its mapped integer) of which we want to calculate items that must appear in its
                recommendation list
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user
        """
        uir_user = test_set.get_user_interactions(user_idx)

        if self._threshold is not None:
            result = pd.unique(uir_user[:, 1][np.where(uir_user[:, 2] >= self._threshold)])
        else:
            # TestRatings just returns the test set of the user
            result = pd.unique(uir_user[:, 1])

        return result.astype(int)


class TestItemsMethodology(Methodology):
    """
    Class which, given a *train set* and a *test set*, has the task to calculate which items must be used in
    order to generate a recommendation list

    With TestItemsMethodology, given a user $u$, items to recommend for $u$ are all items that appear in the
    *test set* of every user excluding those items that appear in the *train set* of $u$

    If the `only_greater_eq` parameter is set, then only items with rating score $>=$ only_greater_eq will be
    returned

    Args:
        only_greater_eq: float which acts as a filter, if specified only items with
            rating score $>=$ only_greater_eq will be returned
    """

    def __init__(self, only_greater_eq: float = None):
        super(TestItemsMethodology, self).__init__(only_greater_eq)

        self._filtered_test_set_items: Optional[np.ndarray] = None

    def __str__(self):
        return "TestItemsMethodology"

    def __repr__(self):
        return f"TestItemsMethodology(only_greater_eq={self._threshold})"

    def setup(self, train_set: Ratings, test_set: Ratings):

        if self._threshold is not None:
            self._filtered_test_set_items = self._filter_only_greater_eq(test_set)
        else:
            self._filtered_test_set_items = test_set.unique_item_idx_column

        self._items_arr = np.arange(len(train_set.item_map))
        self._query_vector = np.zeros(len(test_set.item_map), dtype=bool)

        self._query_vector[self._filtered_test_set_items] = True

        return self

    def filter_single(self, user_idx: int, train_set: Ratings, test_set: Ratings) -> np.ndarray:
        """
        Method that returns items that need to be part of the recommendation list of a single user.
        Since it's the TestItems Methodology, all items that appear in the *test set* of every user will be returned,
        except for those that appear in the *train set* of the user passed as parameter

        Args:
            user_idx: User idx (meaning its mapped integer) of which we want to calculate items that must appear in its
                recommendation list
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user
        """
        already_seen_items_it = pd.unique(train_set.get_user_interactions(user_idx)[:, 1].astype(int))

        self._query_vector[already_seen_items_it] = False

        result = self._items_arr[self._query_vector]

        self._query_vector[self._filtered_test_set_items] = True

        return result.astype(int)


class TrainingItemsMethodology(Methodology):
    """
    Class which, given a *train set* and a *test set*, has the task to calculate which items must be used in
    order to generate a recommendation list

    With TrainingItemsMethodology, given a user $u$, items to recommend for $u$ are all items that appear in the
    'train set' of every user excluding those items that appear in the 'train set' of $u$

    If the `only_greater_eq` parameter is set, then only items with rating score $>=$ only_greater_eq will be
    returned

    Args:
        only_greater_eq: float which acts as a filter, if specified only items with
            rating score $>=$ only_greater_eq will be returned
    """

    def __init__(self, only_greater_eq: float = None):
        super(TrainingItemsMethodology, self).__init__(only_greater_eq)

        self._filtered_train_set_items: Optional[Set] = None

    def __str__(self):
        return "TrainingItemsMethodology"

    def __repr__(self):
        return f"TrainingItemsMethodology(only_greater_eq={self._threshold})"

    def setup(self, train_set: Ratings, test_set: Ratings):

        if self._threshold is not None:
            self._filtered_train_set_items = self._filter_only_greater_eq(train_set)
        else:
            self._filtered_train_set_items = train_set.unique_item_idx_column

        self._items_arr = np.arange(len(train_set.item_map))
        self._query_vector = np.zeros(len(test_set.item_map), dtype=bool)

        self._query_vector[self._filtered_train_set_items] = True

        return self

    def filter_single(self, user_idx: int, train_set: Ratings, test_set: Ratings) -> np.ndarray:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the TrainingItems Methodology, all items that appear in the *train set* of every user will be
        returned, except for those that appear in the *train set* of the user passed as parameter

        Args:
            user_idx: User idx (meaning its mapped integer) of which we want to calculate items that must appear in its
                recommendation list
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user
        """
        already_seen_items_it = pd.unique(train_set.get_user_interactions(user_idx)[:, 1].astype(int))

        self._query_vector[already_seen_items_it] = False

        result = self._items_arr[self._query_vector]

        self._query_vector[self._filtered_train_set_items] = True

        return result.astype(int)


class AllItemsMethodology(Methodology):
    """
    Class which, given a *train set* and a *test set*, has the task to calculate which items must be used in
    order to generate a recommendation list

    With AllItemsMethodology, given a user $u$, items to recommend for $u$ are all items that appear in `items_list`
    parameter excluding those items that appear in the *train set* of $u$

    If `items_list` is None, then the union of items that appear in the train and test set will be considered

    Args:
        items_list: Items set that must appear in the recommendation list of every user. If None, all items that appear
            in the train and test set will be considered
    """

    def __init__(self, items_list: Union[Sequence[str], Sequence[int]] = None):
        self.items_list = items_list
        if items_list is not None:
            self.items_list = np.array(items_list)

        super(AllItemsMethodology, self).__init__(None)

    def __str__(self):
        return "AllItemsMethodology"

    def __repr__(self):
        return f"AllItemsMethodology(items_list={list(self.items_list)})"

    def setup(self, train_set: Ratings, test_set: Ratings):

        if self.items_list is None:
            self.items_list = np.array(list(set(train_set.item_idx_column).union(set(test_set.item_idx_column))))

        elif np.issubdtype(self.items_list.dtype, str):

            masked_items_in_map = train_set.item_map.convert_seq_str2int(self.items_list, missing=-1)
            masked_missing_indices = np.where(masked_items_in_map == -1)
            masked_missing_items = self.items_list[masked_missing_indices]
            train_set.item_map.append(masked_missing_items)
            test_set.item_map.append(masked_missing_items)

            self.items_list = train_set.item_map.convert_seq_str2int(self.items_list)

        self._items_arr = np.arange(len(train_set.item_map))
        self._query_vector = np.zeros(len(test_set.item_map), dtype=bool)

        self._query_vector[self.items_list] = True

        return self

    def filter_single(self, user_idx: int, train_set: Ratings, test_set: Ratings) -> np.ndarray:
        """
        Method that returns items that needs to be part of the recommendation list of a single user.
        Since it's the AllItems Methodology, all items that appear in the `items_list` parameter of the constructor
        will be returned, except for those that appear in the *train set* of the user passed as parameter

        Args:
            user_idx: User idx (meaning its mapped integer) of which we want to calculate items that must appear in its
                recommendation list
            train_set: `Ratings` object which contains the train set of every user
            test_set: `Ratings` object which contains the test set of every user
        """
        already_seen_items_it = pd.unique(train_set.get_user_interactions(user_idx)[:, 1].astype(int))

        self._query_vector[already_seen_items_it] = False
        result = self._items_arr[self._query_vector]
        self._query_vector[self.items_list] = True

        return result.astype(int)

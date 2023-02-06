from __future__ import annotations
from collections import defaultdict

from typing import Set, List, Tuple, TYPE_CHECKING, Union

import abc
from abc import ABC

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Interaction

from clayrs.content_analyzer.ratings_manager.ratings import Ratings
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar


class Partitioning(ABC):
    """
    Abstract class for partitioning technique. Each class must implement the `split_single()` method which specify how
    data for a single user will be split

    Args:
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged when
            calling the `split_all()` method. Otherwise, a `ValueError` exception is raised
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
    def split_single(self, user_ratings: List[Interaction]) -> Tuple[List[List[Interaction]], List[List[Interaction]]]:
        """
        Abstract method in which each partitioning technique must specify how to split data for a single user

        Args:
            user_ratings: List of `Interaction` objects of a single user

        Returns:
            Two lists, where the first contains one list of `Interaction` objects for each split that will
                constitute the *train set* of the user, the second contains one list of `Interaction` objects for each split
                that will constitute the *test set* for the user
        """
        raise NotImplementedError

    def split_all(self, ratings_to_split: Ratings, user_list: Set[int] = None) -> Tuple[List[Ratings], List[Ratings]]:
        """
        Concrete method that splits, for every user in the `ratings_to_split` parameter, the original ratings
        into *train set* and *test set*.
        If a `user_id_list` parameter is set, the method will do the splitting only for the users
        specified inside the list.

        The method returns two lists:

        * The first contains all train set for each split (if the partitioning technique returns more than one split
        e.g. KFold)
        * The second contains all test set for each split (if the partitioning technique returns more than one split
        e.g. KFold)

        Obviously the two lists will have the same length, and to the *train set* in position $i$ corresponds the
        *truth set* at position $i$

        Args:
            ratings_to_split: `Ratings` object which contains the interactions of the users that must be splitted
                into *train set* and *test set*
            user_id_list: The set of users for which splitting will be done. If set, splitting will be performed only
                for users inside the list. Otherwise, splitting will be performed for all users in `ratings_to_split`
                parameter

        Raises:
            ValueError: if `skip_user_error=True` in the constructor and for some users splitting can't be performed
        """

        if user_list is None:
            user_list = ratings_to_split.unique_user_idx_column

        # {
        #   0: {'train': [u1_uir, u2_uir]},
        #       'test': [u1_uir, u2_uir]},
        #
        #   1: {'train': [u1_uir, u2_uir]},
        #       'test': [u1_uir, u2_uir]
        #  }
        train_test_dict = defaultdict(lambda: defaultdict(list))
        count = 0

        with get_progbar(user_list) as pbar:

            pbar.set_description("Performing {}".format(str(self)))
            for user_idx in pbar:
                user_ratings = ratings_to_split.get_user_interactions(user_idx)
                try:
                    user_train_list, user_test_list = self.split_single(user_ratings)
                    for split_number, (single_train, single_test) in enumerate(zip(user_train_list, user_test_list)):
                        # we set for each split the train_set and test_set of every user u1
                        # eg.
                        #     train_test_dict[0]['train']['u1'] = u1_interactions_train0
                        #     train_test_dict[0]['test']['u1'] = u1_interactions_test0
                        # train_test_dict[split_number]['train'][user_id] = single_train
                        # train_test_dict[split_number]['test'][user_id] = single_test
                        train_test_dict[split_number]['train'].extend(single_train)
                        train_test_dict[split_number]['test'].extend(single_test)

                except ValueError as e:
                    if self.skip_user_error:
                        count += 1
                        continue
                    else:
                        raise e

        if count > 0:
            logger.warning(f"{count} users will be skipped because partitioning couldn't be performed\n"
                           f"Change this behavior by setting `skip_user_error` to True")

        train_list = [Ratings.from_uir(np.vstack(train_test_dict[split]['train']),
                                       ratings_to_split.user_map, ratings_to_split.item_map)
                      for split in train_test_dict]

        test_list = [Ratings.from_uir(np.vstack(train_test_dict[split]['test']),
                                      ratings_to_split.user_map, ratings_to_split.item_map)
                     for split in train_test_dict]

        return train_list, test_list


class KFoldPartitioning(Partitioning):
    """
    Class that performs K-Fold partitioning

    Args:
        n_splits (int): Number of splits. Must be at least 2
        shuffle:
            Whether to shuffle the data before splitting into batches.
            Note that the samples within each split will not be shuffled.
        random_state:
            When `shuffle` is True, `random_state` affects the ordering of the
            indices, which controls the randomness of each fold. Otherwise, this
            parameter has no effect.
            Pass an int for reproducible output across multiple function calls.
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged when
            calling the `split_all()` method. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, n_splits: int = 2, shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):
        self.__kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        super(KFoldPartitioning, self).__init__(skip_user_error)

    def split_single(self, uir_user):
        """
        Method which splits in $k$ splits both in *train set* and *test set* the ratings of a single user

        Args:
            user_ratings: List of `Interaction` objects of a single user

        Returns:
            Two lists, where the first contains one list of `Interaction` objects for each split that will
                constitute the *train set* of the user, the second contains one list of `Interaction` objects for each split
                that will constitute the *test set* for the user
        """
        split_result = self.__kf.split(uir_user)

        user_train_list = []
        user_test_list = []
        # split_result contains index of the ratings which must constitutes train set and test set
        for train_set_indexes, test_set_indexes in split_result:
            user_interactions_train = [uir_user[index] for index in train_set_indexes]

            user_interactions_test = [uir_user[index] for index in test_set_indexes]

            user_train_list.append(user_interactions_train)
            user_test_list.append(user_interactions_test)

        return user_train_list, user_test_list

    def __str__(self):
        return "KFoldPartitioning"

    def __repr__(self):
        return f"KFoldPartitioning(n_splits={self.__kf.n_splits}, shuffle={self.__kf.shuffle}, " \
               f"random_state={self.__kf.random_state}, skip_user_error={self.skip_user_error})"


class HoldOutPartitioning(Partitioning):
    """
    Class that performs Hold-Out partitioning

    Args:
        train_set_size: Should be between 0.0 and 1.0 and represent the proportion of the ratings to
            ***hold*** in the train set for each user.
        random_state:
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        shuffle:
            Whether or not to shuffle the data before splitting.
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged when
            calling the `split_all()` method. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, train_set_size: float = 0.8, shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):
        self._check_percentage(train_set_size)
        self.__train_set_size = train_set_size
        self.__test_set_size = (1 - train_set_size)
        self.__random_state = random_state
        self.__shuffle = shuffle

        super().__init__(skip_user_error)

    def split_single(self, uir_user) -> Tuple[List[List[Interaction]], List[List[Interaction]]]:
        """
        Method which splits *train set* and *test set* the ratings of a single user by holding in the train
        set the percentage of data specified in `train_set_size` in the constructor

        Args:
            user_ratings: List of `Interaction` objects of a single user

        Returns:
            Two lists, where the first contains one list of `Interaction` objects that will
                constitute the *train set* of the user, the second contains one list of `Interaction` objects
                that will constitute the *test set* for the user
        """
        uir_train, uir_test = train_test_split(uir_user,
                                               train_size=self.__train_set_size,
                                               test_size=self.__test_set_size,
                                               shuffle=self.__shuffle,
                                               random_state=self.__random_state)

        user_train_list = [uir_train]
        user_test_list = [uir_test]

        return user_train_list, user_test_list

    def __str__(self):
        return "HoldOutPartitioning"

    def __repr__(self):
        return f"HoldOutPartitioning(train_set_size={self.__train_set_size}, shuffle={self.__shuffle}, " \
               f"random_state={self.__random_state}, skip_user_error={self.skip_user_error})"


class BootstrapPartitioning(Partitioning):
    """
    Class that performs Bootstrap Partitioning.

    The bootstrap partitioning consists in executing $n$ extractions with replacement for each user from the original
    interaction frame, where $n$ is the length of the user interactions:

        * The sampled data will be part of the ***train set***
        * All the data which is part of the original dataset but was not sampled will be part of the ***test set***

    The bootstrap partitioning can **change** the original data distribution, since during the extraction phase you
    could sample the same data more than once

    Args:
        random_state:
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged when
            calling the `split_all()` method. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, random_state: int = None, skip_user_error: bool = True):
        super().__init__(skip_user_error)

        self.__random_state = random_state

    def split_single(self, uir_user):
        """
        Method which splits *train set* and *test set* the ratings of a single user by performing $n$ extraction with
        replacement of the user interactions, where $n$ is the number of its interactions.
        The interactions which are not sampled will be part of the *test set*

        Args:
            user_ratings: List of `Interaction` objects of a single user

        Returns:
            Two lists, where the first contains one list of `Interaction` objects that will
                constitute the *train set* of the user, the second contains one list of `Interaction` objects
                that will constitute the *test set* for the user
        """

        interactions_train = resample(uir_user,
                                      replace=True,
                                      n_samples=len(uir_user[:, 0]),
                                      random_state=self.__random_state)

        interactions_test = [interaction
                             for interaction in uir_user
                             if not any(np.array_equal(interaction, interaction_train, equal_nan=True)
                                        for interaction_train in interactions_train)]

        user_train_list = [interactions_train]
        user_test_list = [interactions_test]

        if len(interactions_test) == 0:
            raise ValueError("The test set for the user is empty! Try increasing the number of its interactions!")

        return user_train_list, user_test_list

    def __str__(self):
        return "BootstrapPartitioning"

    def __repr__(self):
        return f"BootstrapPartitioning(random_state={self.__random_state}, skip_user_error={self.skip_user_error})"

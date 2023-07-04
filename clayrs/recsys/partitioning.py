from collections import defaultdict

from typing import Set, List, Tuple, Union

import abc
from abc import ABC

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample

from clayrs.content_analyzer.ratings_manager.ratings import Ratings
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar


class Partitioning(ABC):
    """
    Abstract class for partitioning technique. Each class must implement the `split_single()` method which specify how
    data for a single user will be split

    Args:
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged at the
            end of the split process specifying n° of users skipped. Otherwise, a `ValueError` exception is raised
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
    def split_single(self, uir_user: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Abstract method in which each partitioning technique must specify how to split data for a single user

        Args:
            uir_user: uir matrix containing interactions of a single user

        Returns:
            The first list contains a uir matrix for each split constituting the *train set* of the user

            The second list contains a uir matrix for each split constituting the *test set* of the user
        """
        raise NotImplementedError

    def split_all(self, ratings_to_split: Ratings,
                  user_list: Union[Set[int], Set[str]] = None) -> Tuple[List[Ratings], List[Ratings]]:
        """
        Concrete method that splits, for every user in the user column of `ratings_to_split`, the original ratings
        into *train set* and *test set*.
        If a `user_list` parameter is set, the method will do the splitting only for the users
        specified inside the list (Users can be specified as *strings* or with their mapped *integer*).

        The method returns two lists:

        * The first contains all train set for each split (if the partitioning technique returns more than one split
        e.g. KFold)
        * The second contains all test set for each split (if the partitioning technique returns more than one split
        e.g. KFold)

        Obviously the two lists will have the same length, and to the *train set* in position $i$ corresponds the
        *truth set* at position $i$

        Args:
            ratings_to_split: `Ratings` object which contains the interactions of the users that must be split
                into *train set* and *test set*
            user_list: The Set of users for which splitting will be done. If set, splitting will be performed only
                for users inside the list. Otherwise, splitting will be performed for all users in `ratings_to_split`
                parameter. User can be specified with their string id or with their mapped integer

        Raises:
            ValueError: if `skip_user_error=True` in the constructor and for at least one user splitting
                can't be performed
        """

        # convert user list to list of int if necessary (strings are passed)
        if user_list is not None:
            all_users = np.array(list(user_list))
            if np.issubdtype(all_users.dtype, str):
                all_users = ratings_to_split.user_map.convert_seq_str2int(all_users)

            all_users = set(all_users)
        else:
            all_users = set(ratings_to_split.unique_user_idx_column)

        # {
        #   0: {'train': [u1_uir, u2_uir]},
        #       'test': [u1_uir, u2_uir]},
        #
        #   1: {'train': [u1_uir, u2_uir]},
        #       'test': [u1_uir, u2_uir]
        #  }
        train_test_dict = defaultdict(lambda: defaultdict(list))
        error_count = 0

        with get_progbar(all_users) as pbar:

            pbar.set_description("Performing {}".format(str(self)))
            for user_idx in pbar:
                user_ratings = ratings_to_split.get_user_interactions(user_idx)
                try:
                    user_train_list, user_test_list = self.split_single(user_ratings)
                    for split_number, (single_train, single_test) in enumerate(zip(user_train_list, user_test_list)):

                        train_test_dict[split_number]['train'].append(single_train)
                        train_test_dict[split_number]['test'].append(single_test)

                except ValueError as e:
                    if self.skip_user_error:
                        error_count += 1
                        continue
                    else:
                        raise e from None

        if error_count > 0:
            logger.warning(f"{error_count} users will be skipped because partitioning couldn't be performed\n"
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
            If set to True, users for which data can't be split will be skipped and only a warning will be logged at the
            end of the split process specifying n° of users skipped. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, n_splits: int = 2, shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):
        self.__kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        super(KFoldPartitioning, self).__init__(skip_user_error)

    def split_single(self, uir_user: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Method which splits in $k$ splits both in *train set* and *test set* the ratings of a single user

        Args:
            uir_user: uir matrix containing interactions of a single user

        Returns:
            The first list contains a uir matrix for each split constituting the *train set* of the user

            The second list contains a uir matrix for each split constituting the *test set* of the user
        """
        split_result = self.__kf.split(uir_user)

        user_train_list = []
        user_test_list = []

        # split_result contains index of the ratings which must constitutes train set and test set
        for train_set_indexes, test_set_indexes in split_result:
            user_interactions_train = uir_user[train_set_indexes]
            user_interactions_test = uir_user[test_set_indexes]

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
            ***hold*** in the train set for each user. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
        test_set_size: If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If `train_size` is also None, it will
            be set to 0.25.
        random_state:
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        shuffle:
            Whether to shuffle the data before splitting.
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged at the
            end of the split process specifying n° of users skipped. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, train_set_size: Union[float, int, None] = None, test_set_size: Union[float, int, None] = None,
                 shuffle: bool = True, random_state: int = None,
                 skip_user_error: bool = True):

        if train_set_size is not None and train_set_size < 0:
            raise ValueError("train_set_size must be a positive number")

        if test_set_size is not None and test_set_size < 0:
            raise ValueError("test_set_size must be a positive number")

        if isinstance(train_set_size, float) and train_set_size > 1.0:
            raise ValueError("train_set_size must be between 0.0 and 1.0")

        if isinstance(test_set_size, float) and test_set_size > 1.0:
            raise ValueError("test_set_size must be between 0.0 and 1.0")

        if isinstance(train_set_size, float) and isinstance(test_set_size, float) and \
                (train_set_size + test_set_size) > 1.0:
            raise ValueError("train_set_size and test_set_size percentages must not sum to a value greater than 1.0")

        self.__train_set_size = train_set_size
        self.__test_set_size = test_set_size
        self.__random_state = random_state
        self.__shuffle = shuffle

        super().__init__(skip_user_error)

    def split_single(self, uir_user: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Method which splits *train set* and *test set* the ratings of a single user by holding in the train set of the
        user interactions accoring to the parameters set in the constructor

        Args:
            uir_user: uir matrix containing interactions of a single user

        Returns:
            The first list contains a uir matrix for each split constituting the *train set* of the user

            The second list contains a uir matrix for each split constituting the *test set* of the user
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
        return f"HoldOutPartitioning(train_set_size={self.__train_set_size}, test_set_size={self.__test_set_size}, " \
               f"shuffle={self.__shuffle}, random_state={self.__random_state}, skip_user_error={self.skip_user_error})"


class BootstrapPartitioning(Partitioning):
    """
    Class that performs Bootstrap Partitioning.

    The bootstrap partitioning consists in executing $n$ extractions with replacement for each user from the original
    interaction frame, where $n$ is the length of the user interactions:

    * The sampled data will be part of the ***train set***
    * All the data which is part of the original dataset but was not sampled will be part of the ***test set***

    !!! info

        The bootstrap partitioning can **change** the original data distribution, since during the extraction phase you
        could sample the same data more than once

    Args:
        random_state:
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
        skip_user_error:
            If set to True, users for which data can't be split will be skipped and only a warning will be logged at the
            end of the split process specifying n° of users skipped. Otherwise, a `ValueError` exception is raised
    """

    def __init__(self, random_state: int = None, skip_user_error: bool = True):
        super().__init__(skip_user_error)

        self.__random_state = random_state

    def split_single(self, uir_user: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Method which splits *train set* and *test set* the ratings of a single user by performing $n$ extraction with
        replacement of the user interactions, where $n$ is the number of its interactions.
        The interactions which are not sampled will be part of the *test set*

        Args:
            uir_user: uir matrix containing interactions of a single user

        Returns:
            The first list contains a uir matrix for each split constituting the *train set* of the user

            The second list contains a uir matrix for each split constituting the *test set* of the user
        """

        interactions_train = resample(uir_user,
                                      replace=True,
                                      n_samples=len(uir_user[:, 0]),
                                      random_state=self.__random_state)

        interactions_test = np.array([interaction
                                      for interaction in uir_user
                                      if not any(np.array_equal(interaction, interaction_train, equal_nan=True)
                                                 for interaction_train in interactions_train)])

        user_train_list = [interactions_train]
        user_test_list = [interactions_test]

        if len(interactions_test) == 0:
            raise ValueError("The test set for the user is empty! Try increasing the number of its interactions!")

        return user_train_list, user_test_list

    def __str__(self):
        return "BootstrapPartitioning"

    def __repr__(self):
        return f"BootstrapPartitioning(random_state={self.__random_state}, skip_user_error={self.skip_user_error})"

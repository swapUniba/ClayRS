from __future__ import annotations
import functools
import os
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Iterator, TYPE_CHECKING, Tuple, Sequence, Callable

import pandas as pd
import numpy_indexed as npi

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.score_processor import ScoreProcessor
    from clayrs.content_analyzer.raw_information_source import RawInformationSource

from clayrs.content_analyzer.exceptions import handler_score_not_float, handler_empty_matrix, UserNone, ItemNone
from clayrs.utils.context_managers import get_progbar
from clayrs.utils.save_content import get_valid_filename


class StrIntMap:
    """
    Class responsible to keep track of the mapping between string ids (from the original source) and integer ids
    (created by the framework and used instead of the string original ones)
    This conversion is done for performance reasons, because accessing contents based on an integer id is
    faster when compared to access using a string id, and also algorithms usually reason based on integer
    representations for users and items (e.g. VBPR)

    The mapping is organized using a numpy array containing string elements:

    ```title="Mapping example visualized"
    ["i1", "i2", "i3", "i4", "i5", ...]
    ```

    The corresponding mapping is obtained by the index that each string id has inside this array:

    ```title="Mapping example visualized: integer mapping"
    +---------+---------+
    | idx     | item_id |
    +---------+---------+
    | 0       | i1      |
    | 1       | i2      |
    | 2       | i3      |
    | 3       | i4      |
    | 4       | i5      |
    +---------+---------+
    ```

    The mapping is defined as input, and it can be one of three different types:

        - Dict[str, int]: the dictionary should have string ids as keys and their mapping to integers as values.
            Mapped integers should start from 0 and end with a number X, where X is the number of elements in the
            dictionary and should have no missing values in between. Otherwise, a LookupError exception will be
            raised;

            ex: {"i1": 0, "i2": 1, "i3": 2, ...}

        - np.ndarray: the numpy array should contain the elements to map;

            ex: ["i1", "i2", "i3", ...]

        - StrIntMap: the same mapping will be used.

    Args:
        str_int_map: object containing the mapping between string ids and integers, it can be a dictionary, an array or
            another StrIntMap object.
    """

    def __init__(self, str_int_map: Union[Dict[str, int], np.ndarray, StrIntMap]):

        if isinstance(str_int_map, dict):
            # dictionary should contain all numbers starting from 0 without holes
            sorted_str = []
            sorted_int = []
            try:
                for str_id, int_idx in sorted(str_int_map.items(), key=lambda item: item[1]):
                    sorted_str.append(str(str_id))
                    sorted_int.append(int(int_idx))
            except ValueError:
                raise LookupError("Mapping dictionary not in the right format! Strings must be mapped to "
                                  "integers starting from 0 without holes!") from None

            if sorted_int != list(range(len(str_int_map))):
                raise LookupError("Mapping dictionary not in the right format! Strings must be mapped to "
                                  "integers starting from 0 without holes!")

            self.map = np.array(sorted_str)
        elif isinstance(str_int_map, np.ndarray):
            self.map = str_int_map.astype(str)
        elif isinstance(str_int_map, StrIntMap):
            self.map = str_int_map.map

    def convert_seq_int2str(self, idx_list: Sequence[int]) -> np.ndarray:
        """
        Method to convert a sequence of integers to the corresponding string ids in the mapping.
        In case of an empty sequence, an empty array will be returned.

        If the integer is not present in the mapping, a IndexError exception is raised.

        Args:
            idx_list: sequence object containing integers to convert
        """
        return self.map[idx_list] if len(idx_list) else np.array([], dtype=str)

    def convert_seq_str2int(self, id_list: Sequence[str], missing="raise") -> np.ndarray:
        """
        Method to convert a sequence of strings to the corresponding integer ids in the mapping.
        In case of an empty sequence, an empty array will be returned.

        If a string is not present in the mapping, a KeyError exception is raised. This behavior can be changed by
        modifying the "missing" parameter.

        Args:
            id_list: sequence object containing strings to convert
            missing: defines the behavior of the method in case a string is not present in the mapping. The possible
                values for this parameter can be checked [here](https://github.com/EelcoHoogendoorn/Numpy_arraysetops_EP/blob/master/numpy_indexed/arraysetops.py#L115)
        """
        return npi.indices(self.map, id_list, missing=missing) if len(id_list) else np.array([], dtype=int)

    def convert_int2str(self, idx: int) -> str:
        """
        Method to convert a single integer to the corresponding string id in the mapping.

        If the integer is not present in the mapping, a IndexError exception is raised.

        Args:
            idx: integer to convert
        """
        return self.map[idx]

    def convert_str2int(self, id: str) -> int:
        """
        Method to convert a single string to the corresponding integer id in the mapping.

        If the string is not present in the mapping, a IndexError exception is raised.

        Args:
            id: string to convert
        """
        # first [0] because np.where returns a tuple, second [0] to access first element
        return np.where(self.map == id)[0][0]

    def append(self, ids_str_to_append: Union[Sequence[str], str]):
        """
        Method to append new string ids to the mapping.

        Note that no control is carried out to check that the new ids do not already exist in the mapping.

        Example:

            >>> mapping = StrIntMap(np.array(["i1", "i2", "i3"]))

            ```
            +---------+---------+
            | idx     | item_id |
            +---------+---------+
            | 0       | i1      |
            | 1       | i2      |
            | 2       | i3      |
            +---------+---------+
            ```

            >>> mapping.append(["i4", "i5"])

            ```
            +---------+---------+
            | idx     | item_id |
            +---------+---------+
            | 0       | i1      |
            | 1       | i2      |
            | 2       | i3      |
            | 3       | i4      |
            | 4       | i5      |
            +---------+---------+
            ```

        Args:
            ids_str_to_append: sequence object containing strings (or single string) to append to the mapping
        """
        self.map = np.hstack((self.map, ids_str_to_append))

    def to_dict(self):
        """
        Method to convert the mapping from its numpy representation to a dictionary based one.

        The resulting dictionary will have the following form:

        ```title="Mapping example visualized: array to dictionary"
        {
            "i1": 0,
            "i2": 1,
            "i3": 2,
            "i4": 3,
            "i5": 4,
            ...
        }
        ```

        Note that the dictionary representation might occupy a lot of space in memory depending on the
        size of the mapping
        """
        return {str_idx: int_idx for int_idx, str_idx in enumerate(self.map)}

    @staticmethod
    def _check_bound_str(f: Callable, item: Union[str, Sequence[str]]):
        # used to replace an exception raised when the given strings (or string) don't exist in the mapping
        # this method is used for the __getitem__ magic method
        try:
            return f(item)
        except (KeyError, IndexError):
            if isinstance(item, str):
                raise KeyError(f"Item {item} not present in the mapping!") from None
            else:
                raise KeyError(f"One or more item used as indices are not present in the mapping!") from None

    def _check_bound_int(self, f: Callable, item: Union[int, Sequence[int]]):
        # used to replace an exception raised when the given integers (or integer) don't exist in the mapping
        # this method is used for the __getitem__ magic method
        try:
            return f(item)
        except IndexError:
            if isinstance(item, int):
                raise IndexError(f"Index {item} is out of bounds for mapping with len={len(self)}") from None
            else:
                raise IndexError(f"One or more indices is out of bounds for mapping with len={len(self)}") from None

    # very ugly but very user-friendly: this is only for the end user!!! Inside the framework you should
    # never call it, it is slow!!.
    # Please directly use one of the above (convert_int2str, convert_seq_str2int, etc.)
    def __getitem__(self, item: Union[Sequence[int], Sequence[str], int, str]) -> Union[np.ndarray[int],
                                                                                        np.ndarray[str],
                                                                                        int, str]:
        """
        This magic method allows to specify any of the accepted type of inputs (that is, sequences or single elements of
        type string and integer) and obtain the corresponding matching output (string output if the input is integer,
        integer output if the input is string)

        The method is slow since it needs to perform different checks, it shouldn't be used internally
        """
        if isinstance(item, str):
            return self._check_bound_str(self.convert_str2int, item)

        elif isinstance(item, int):
            return self._check_bound_int(self.convert_int2str, item)

        elif isinstance(item, Sequence) or (isinstance(item, np.ndarray) and np.squeeze(item).ndim <= 1):
            arr_iterable = np.array(item)

            if np.issubdtype(arr_iterable.dtype, str):
                return self._check_bound_str(self.convert_seq_str2int, arr_iterable)
            elif np.issubdtype(arr_iterable.dtype, int):
                return self._check_bound_int(self.convert_seq_int2str, arr_iterable)
            else:
                raise TypeError("Iterable to convert should only contains numbers or strings!")

        else:
            raise TypeError("Item not supported! You should use a sequence of str/int or a scalar str/int! "
                            "Try using a list or a 1d array")

    def __len__(self):
        return len(self.map)

    def __repr__(self):
        return repr(self.to_dict())

    def __iter__(self):
        yield from self.map

    def __hash__(self):
        return hash(self.map.tostring())

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return np.array_equal(self.map, other.map)
        else:
            return False


class Ratings:
    """
    Class responsible for importing an interaction frame into the framework

    **If** the source file contains users, items and ratings in this order,
    no additional parameters are needed, **otherwise** the mapping must be explicitly specified using:

    * **'user_id'** column,
    * **'item_id'** column,
    * **'score'** column

    The *score* column can also be processed: in case you would like to consider as score the sentiment of a textual
    review, or maybe normalizing all scores in $[0, 1]$ range. Check the example below for more

    Note that, during the import phase, the user and item ids will be converted to integers and a mapping between
    the newly created ids and the original string ids will be created. For replicability purposes, it is possible to
    pass your custom item and user map instead of leaving this task to the framework. Check the example below to see how

    Examples:

        ```title="CSV raw source"
        user_id,item_id,rating,timestamp,review
        u1,i1,4,00112,good movie
        u2,i1,3,00113,an average movie
        u2,i32,2,00114,a bad movie
        ```

        As you can see the user id column, item id column and score column are the first three column and are already in
        sequential order, so no additional parameter is required to the Ratings class:

        >>> import clayrs.content_analyzer as ca
        >>> ratings_raw_source = ca.CSVFile('ratings.csv')
        >>> # add timestamp='timestamp' to the following if
        >>> # you want to load also the timestamp
        >>> ratings = ca.Ratings(ratings_raw_source)

        In case columns in the raw source are not in the above order you must specify an appropriate mapping via
        positional index (useful in case your raw source doesn't have a header) or via column ids:

        >>> # (mapping by index) EQUIVALENT:
        >>> ratings = ca.Ratings(
        >>> ca.CSVFile('ratings.csv'),
        >>> user_id_column=0,  # (1)
        >>> item_id_column=1,  # (2)
        >>> score_column=2  # (3)
        >>> )

        1. First column of raw source is the column containing all user ids
        2. Second column of raw source is the column containing all item ids
        3. Third column of raw source is the column containing all the scores

        >>> # (mapping by column name) EQUIVALENT:
        >>> ratings = ca.Ratings(
        >>> ca.CSVFile('ratings.csv'),
        >>> user_id_column='user_id',  # (1)
        >>> item_id_column='item_id',  # (2)
        >>> score_column='rating'  # (3)
        >>> )

        1. The column with id 'user_id' of raw source is the column containing all user ids
        2. The column with id 'item_id' of raw source is the column containing all item ids
        3. The column with id 'rating' of raw source is the column containing all the scores


        In case you would like to use the sentiment of the `review` column of the above raw source as score column,
        simply specify the appropriate `ScoreProcessor` object

        >>> ratings_raw_source = ca.CSVFile('ratings.csv')
        >>> ratings = ca.Ratings(ratings_raw_source,
        >>>                      score_column='review',
        >>>                      score_processor=ca.TextBlobSentimentAnalysis())



        In case you would like to specify the mappings for items or users, simply specify them in the corresponding
        parameters

        >>> ratings_raw_source = ca.CSVFile('ratings.csv')
        >>> custom_item_map = {'i1': 0, 'i2': 2, 'i3': 1}
        >>> custom_user_map = {'u1': 0, 'u2': 2, 'u3': 1}
        >>> ratings = ca.Ratings(ratings_raw_source,
        >>>                      item_map=custom_item_map,
        >>>                      user_map=custom_user_map)

    Args:
        source: Source containing the raw interaction frame
        user_id_column: Name or positional index of the field of the raw source representing *users* column
        item_id_column: Name or positional index of the field of the raw source representing *items* column
        score_column: Name or positional index of the field of the raw source representing *score* column
        timestamp_column: Name or positional index of the field of the raw source representing *timesamp* column
        score_processor: `ScoreProcessor` object which will process the `score_column` accordingly. Useful if you want
            to perform sentiment analysis on a textual column or you want to normalize all scores in $[0, 1]$ range
        item_map: dictionary with string keys (the item ids) and integer values (the corresponding unique integer ids)
            used to create the item mapping. If not specified, it will be automatically created internally
        user_map: dictionary with string keys (the user ids) and integer values (the corresponding unique integer ids)
            used to create the user mapping. If not specified, it will be automatically created internally
    """

    def __init__(self, source: RawInformationSource,
                 user_id_column: Union[str, int] = 0,
                 item_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: ScoreProcessor = None,
                 item_map: Dict[str, int] = None,
                 user_map: Dict[str, int] = None):

        # utility dictionary that will contain each user index as key and a numpy array, containing the indexes of the
        # rows in the uir matrix which refer to an interaction for that user, as value. This is done to optimize
        # performance when requesting all interactions of a certain user
        self._user2rows: Dict

        self._uir: np.ndarray
        self.item_map: StrIntMap
        self.user_map: StrIntMap

        self._import_ratings(source, user_id_column, item_id_column,
                             score_column, timestamp_column, score_processor, item_map, user_map)

    @property
    def uir(self) -> np.ndarray:
        """
        Getter for the uir matrix created from the interaction frame.
        The imported ratings are converted in the form of a numpy ndarray where each row will represent an interaction.
        This uir matrix can be seen in a tabular representation as follows:

        ```title="UIR matrix visualized: tabular format"
        +----------+----------+--------+-----------+
        | user_idx | item_idx | score  | timestamp |
        +----------+----------+--------+-----------+
        | 0.       | 0.       | 4      | np.nan    |
        | 0.       | 1.       | 3      | np.nan    |
        | 1.       | 4.       | 1      | np.nan    |
        +----------+----------+--------+-----------+
        ```

        Where the 'user_idx' and 'item_idx' columns contain the integer ids from the mapping of the
        `Ratings` object itself (these integer ids match the string ids that are in the original interaction frame)
        """
        return self._uir

    @functools.cached_property
    @handler_empty_matrix(dtype=int)
    def user_idx_column(self) -> np.ndarray:
        """
        Getter for the 'user_idx' column of the uir matrix. This will return the user column "as is", so it will contain
        duplicate users. Use the 'unique_user_idx_column' method to get unique users.

        Returns:
            Users column with duplicates (integer ids)
        """
        return self._uir[:, 0].astype(int)

    @functools.cached_property
    def unique_user_idx_column(self) -> np.ndarray:
        """
        Getter for the 'user_idx' column of the uir matrix. This will return the user column without duplicates.

        Returns:
            Users column without duplicates (integer ids)
        """
        return pd.unique(self.user_idx_column)

    @functools.cached_property
    def user_id_column(self) -> np.ndarray:
        """
        Getter for the 'user_id' column of the interaction frame. This will return the user column "as is", so it will
        contain duplicate users. Use the 'unique_user_id_column' method to get unique users.

        Returns:
            Users column with duplicates (string ids)
        """
        return self.user_map.convert_seq_int2str(self.user_idx_column)

    @functools.cached_property
    def unique_user_id_column(self) -> np.ndarray:
        """
        Getter for the 'user_id' column of the interaction frame. This will return the user column without duplicates.

        Returns:
            Users column without duplicates (string ids)
        """
        return self.user_map.convert_seq_int2str(self.unique_user_idx_column)

    @functools.cached_property
    @handler_empty_matrix(dtype=int)
    def item_idx_column(self) -> np.ndarray:
        """
        Getter for the 'item_idx' column of the uir matrix. This will return the item column "as is", so it will contain
        duplicate items. Use the 'unique_item_idx_column' method to get unique items.

        Returns:
            Items column with duplicates (integer ids)
        """
        return self._uir[:, 1].astype(int)

    @functools.cached_property
    def unique_item_idx_column(self) -> np.ndarray:
        """
        Getter for the 'item_idx' column of the uir matrix. This will return the item column without duplicates.

        Returns:
            Items column without duplicates (integer ids)
        """
        return pd.unique(self.item_idx_column)

    @functools.cached_property
    def item_id_column(self) -> np.ndarray:
        """
        Getter for the 'item_id' column of the interaction frame. This will return the item column "as is", so it will
        contain duplicate items. Use the 'unique_item_id_column' method to get unique items.

        Returns:
            Items column with duplicates (string ids)
        """
        return self.item_map.convert_seq_int2str(self.item_idx_column)

    @functools.cached_property
    def unique_item_id_column(self) -> np.ndarray:
        """
        Getter for the 'item_id' column of the interaction frame. This will return the item column without duplicates.

        Returns:
            Items column without duplicates (string ids)
        """
        return self.item_map.convert_seq_int2str(self.unique_item_idx_column)

    @functools.cached_property
    @handler_empty_matrix(dtype=float)
    def score_column(self) -> np.ndarray:
        """
        Getter for the score column. This will return the score column "as is".

        Returns:
            Score column
        """
        return self._uir[:, 2].astype(float)

    @functools.cached_property
    @handler_empty_matrix(dtype=int)
    def timestamp_column(self) -> np.ndarray:
        """
        Getter for the timestamp column. This will return the score column "as is". If no timestamp is present then an
        empty list is returned

        Returns:
            Timestamp column or empty list if no timestamp is present
        """
        return self._uir[:, 3][~np.isnan(self._uir[:, 3])].astype(int)

    @handler_score_not_float
    def _import_ratings(self, source: RawInformationSource,
                        user_column: Union[str, int],
                        item_column: Union[str, int],
                        score_column: Union[str, int],
                        timestamp_column: Union[str, int],
                        score_processor: ScoreProcessor,
                        item_map: Dict[str, int],
                        user_map: Dict[str, int]):
        """
        This method is used internally to create the uir matrix from the interaction frame
        """

        # lists that will contain the original data temporarily
        # this is so that the conversion from string ids to integers will be called only once
        # said lists will also be used to create the mappings if not specified in the parameters
        tmp_user_id_column = []
        tmp_item_id_column = []
        tmp_score_column = []
        tmp_timestamp_column = []

        with get_progbar(source) as pbar:

            pbar.set_description(desc="Importing ratings")
            for i, row in enumerate(pbar):

                user_id = self._get_field_data(user_column, row)
                item_id = self._get_field_data(item_column, row)
                score = self._get_field_data(score_column, row)
                timestamp = np.nan

                if score_processor is not None:
                    score = score_processor.fit(score)
                else:
                    score = float(score)

                if timestamp_column is not None:
                    timestamp = int(self._get_field_data(timestamp_column, row))

                tmp_user_id_column.append(user_id)
                tmp_item_id_column.append(item_id)
                tmp_score_column.append(score)
                tmp_timestamp_column.append(timestamp)

        # create the item_map from the item_id column if not specified
        if item_map is None:
            self.item_map = StrIntMap(np.array(list(dict.fromkeys(tmp_item_id_column))))
        else:
            self.item_map = StrIntMap(item_map)

        # create the user_map from the user_id column if not specified
        if user_map is None:
            self.user_map = StrIntMap(np.array(list(dict.fromkeys(tmp_user_id_column))))
        else:
            self.user_map = StrIntMap(user_map)

        # convert user and item ids and create the uir matrix
        self._uir = np.array((
            self.user_map.convert_seq_str2int(np.array(tmp_user_id_column)),
            self.item_map.convert_seq_str2int(np.array(tmp_item_id_column)),
            tmp_score_column, tmp_timestamp_column
        )).T

        # create the utility dictionary user2rows
        self._user2rows = {
            user_idx: np.where(self._uir[:, 0] == user_idx)[0]
            for user_idx in self.unique_user_idx_column
        }

    def get_user_interactions(self, user_idx: int, head: int = None, as_indices: bool = False) -> np.ndarray:
        """
        Method which returns a two-dimensional numpy array containing all the rows from the uir matrix for a single
        user, one for each interaction of the user.
        Then you can easily access the columns of the resulting array to obtain useful information

        Examples:

            So if the rating frame is the following:

            ```
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u1      | i2      |     3 |
            | u2      | i5      |     1 |
            +---------+---------+-------+
            ```

            The corresponding uir matrix will be the following:

            ```
            +----------+----------+--------+-----------+
            | user_idx | item_idx | score  | timestamp |
            +----------+----------+--------+-----------+
            | 0.       | 0.       | 4      | np.nan    |
            | 0.       | 1.       | 3      | np.nan    |
            | 1.       | 4.       | 1      | np.nan    |
            +----------+----------+--------+-----------+
            ```

            >>> rating_frame.get_user_interactions(0)
            np.ndarray([
                [0. 0. 4 np.nan],
                [0. 1. 3 np.nan],
            ])

            So you could easily extract all the ratings that a user has given, for example:

            >>> rating_frame.get_user_interactions(0)[:, 2]
            np.ndarray([4,
                        3])

            If you only want the first $k$ interactions of the user, set `head=k`. The interactions returned are the
            first $k$ according to their order of appearance in the rating frame:

            >>> rating_frame.get_user_interactions(0, head=1)
            np.ndarray([
                [0. 0. 4 np.nan]
            ])

            If you want to have the indices of the uir matrix corresponding to the user interactions instead of the
            actual interactions, set `as_indices=True`. This will return a numpy array containing the indexes of
            the rows of the uir matrix for the interactions of the specified user

            >>> rating_frame.get_user_interactions(0, as_indices=True)
            np.ndarray([0, 1])

            If you don't know the `user_idx` for a specific user, you can obtain it using the user map as follows:

            >>> user_idx = rating_frame.user_map['u1']
            >>> rating_frame.get_user_interactions(user_idx=user_idx)
            np.ndarray([
                [0. 0. 4 np.nan],
                [0. 1. 3 np.nan],
            ])

        Args:
            user_idx: Integer id of the user for which you want to retrieve the interactions
            head: Integer which will cut the list of interactions of the user returned. The interactions returned are
                the first $k$ according to their order of appearance
            as_indices: Instead of returning the user interactions, the indices of the rows in the uir matrix
                corresponding to interactions for the specified user will be returned

        Returns:
            If `as_indices=False`, numpy ndarray containing the rows from the uir matrix for the specified user,
                otherwise numpy array containing the indexes of the rows from the uir matrix for the interactions of the
                specified user

        """
        user_rows = self._user2rows.get(user_idx, [])[:head]
        return user_rows if as_indices else self._uir[user_rows]

    def filter_ratings(self, user_list: Sequence[int]) -> Ratings:
        """
        Method which will filter the rating frame by keeping only interactions of users appearing in the `user_list`.
        This method will return a new `Ratings` object without changing the original

        Examples:

            ```title="Starting Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u1      | i2      |     3 |
            | u2      | i5      |     1 |
            +---------+---------+-------+
            ```

            ```title="Starting Rating object: corresponding uir matrix"
            +----------+----------+--------+-----------+
            | user_idx | item_idx | score  | timestamp |
            +----------+----------+--------+-----------+
            | 0.       | 0.       | 4      | np.nan    |
            | 0.       | 1.       | 3      | np.nan    |
            | 1.       | 4.       | 1      | np.nan    |
            +----------+----------+--------+-----------+
            ```

            >>> rating_frame.filter_ratings([0])

            ```title="Returned Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u1      | i2      |     3 |
            +---------+---------+-------+
            ```

            ```title="Returned Rating object: corresponding uir matrix"
            +----------+----------+--------+-----------+
            | user_idx | item_idx | score  | timestamp |
            +----------+----------+--------+-----------+
            | 0.       | 0.       | 4      | np.nan    |
            | 0.       | 1.       | 3      | np.nan    |
            +----------+----------+--------+-----------+
            ```

            If you don't know the integer ids for the users, you can obtain them using the user map as follows:

            >>> user_idxs = rating_frame.user_map[['u1']]
            >>> rating_frame.filter_ratings(user_list=user_idxs)

        Args:
            user_list: List of user integer ids that will be present in the filtered `Ratings` object

        Returns
            The filtered Ratings object which contains only interactions of selected users
        """
        valid_indexes = np.where(np.isin(self.user_idx_column, user_list))
        new_uir = self._uir[valid_indexes]

        return Ratings.from_uir(new_uir, self.user_map.map, self.item_map.map)

    def take_head_all(self, head: int) -> Ratings:
        """
        Method which will retain only $k$ interactions for each user. The $k$ interactions retained are the first which
        appear in the rating frame.

        This method will return a new `Ratings` object without changing the original

        Examples:

            ```title="Starting Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u1      | i2      |     3 |
            | u2      | i5      |     1 |
            | u2      | i6      |     2 |
            +---------+---------+-------+
            ```

            ```title="Starting Rating object: corresponding uir matrix"
            +----------+----------+--------+-----------+
            | user_idx | item_idx | score  | timestamp |
            +----------+----------+--------+-----------+
            | 0.       | 0.       | 4      | np.nan    |
            | 0.       | 1.       | 3      | np.nan    |
            | 1.       | 4.       | 1      | np.nan    |
            | 1.       | 5.       | 2      | np.nan    |
            +----------+----------+--------+-----------+
            ```

            >>> rating_frame.take_head_all(head=1)

            ```title="Returned Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u2      | i5      |     1 |
            +---------+---------+-------+
            ```

            ```title="Returned Rating object: corresponding uir matrix"
            +----------+----------+--------+-----------+
            | user_idx | item_idx | score  | timestamp |
            +----------+----------+--------+-----------+
            | 0.       | 0.       | 4      | np.nan    |
            | 1.       | 4.       | 1      | np.nan    |
            +----------+----------+--------+-----------+
            ```

        Args:
            head: The number of interactions to retain for each user

        Returns:
            The filtered Ratings object which contains only first $k$ interactions for each user
        """
        cut_rows = np.hstack((rows[:head] for rows in self._user2rows.values()))
        new_uir = self._uir[cut_rows]

        return Ratings.from_uir(new_uir, self.user_map.map, self.item_map.map)

    def to_dataframe(self, ids_as_str: bool = True) -> pd.DataFrame:
        """
        Method which will convert the `Rating` object to a `pandas DataFrame object`.

        The returned DataFrame object will contain the 'user_id', 'item_id' and 'score' column and optionally the
        'timestamp' column, if at least one interaction has a timestamp.

        Args:
            ids_as_str: If True, the original string ids for users and items will be used, otherwise their integer ids

        Returns:
            The rating frame converted to a pandas DataFrame with 'user_id', 'item_id', 'score' column and optionally
                the 'timestamp' column

        """
        if ids_as_str:
            will_be_frame = {'user_id': self.user_id_column,
                             'item_id': self.item_id_column,
                             'score': self.score_column}
        else:
            will_be_frame = {'user_id': self.user_idx_column,
                             'item_id': self.item_idx_column,
                             'score': self.score_column}

        if len(self.timestamp_column) != 0:
            will_be_frame['timestamp'] = self.timestamp_column

        return pd.DataFrame(will_be_frame)

    def to_csv(self, output_directory: str = '.', file_name: str = 'ratings_frame', overwrite: bool = False,
               ids_as_str: bool = True):
        """
        Method which will save the `Ratings` object to a `csv` file

        Args:
            output_directory: directory which will contain the csv file
            file_name: Name of the csv_file
            overwrite: If set to True and a csv file exists in the same output directory with the same file name, it
                will be overwritten
            ids_as_str: If True the original string ids for users and items will be used, otherwise their integer ids
        """
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        file_name = get_valid_filename(output_directory, file_name, 'csv', overwrite)

        frame = self.to_dataframe(ids_as_str=ids_as_str)
        frame.to_csv(os.path.join(output_directory, file_name), index=False, header=True)

    @staticmethod
    def _get_field_data(field_name: Union[str, int], row: Dict):
        try:
            if isinstance(field_name, str):
                data = row[field_name]
            else:
                row_keys = list(row.keys())
                key = row_keys[field_name]
                data = row[key]

        except KeyError:
            raise KeyError("Column {} not found in the raw source".format(field_name))
        except IndexError:
            raise IndexError("Column index {} not present in the raw source".format(field_name))

        return str(data)

    @classmethod
    @handler_score_not_float
    def from_dataframe(cls, interaction_frame: pd.DataFrame,
                       user_column: Union[str, int] = 0,
                       item_column: Union[str, int] = 1,
                       score_column: Union[str, int] = 2,
                       timestamp_column: Union[str, int] = None,
                       user_map: Union[Dict[str, int], np.ndarray, StrIntMap] = None,
                       item_map: Union[Dict[str, int], np.ndarray, StrIntMap] = None) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing pandas DataFrame

        **If** the pandas DataFrame contains users, items and ratings in this order,
        no additional parameters are needed, **otherwise** the mapping must be explicitly specified using:

        * **'user_id'** column,
        * **'item_id'** column,
        * **'score'** column

        Check documentation of the `Ratings` class for examples on mapping columns explicitly, the functioning is the
        same

        Furthermore, it is also possible to specify the user and item mapping between original string ids and
        integer ones. However, differently from the `Ratings` class documentation, it is possible not only to specify
        them as dictionaries but also as numpy arrays or `StrIntMap` objects directly. The end result will be the same
        independently of the type, but it is suggested to check the `StrIntMap` class documentation to understand
        the differences between the three possible types

        Examples:

            >>> ratings_df = pd.DataFrame({'user_id': ['u1', 'u1', 'u1'],
            >>>                            'item_id': ['i1', 'i2', 'i3'],
            >>>                            'score': [4, 3, 3])
            >>> Ratings.from_dataframe(ratings_df)

            or

            >>> user_map = {'u1': 0}
            >>> item_map = {'i1': 0, 'i2': 2, 'i3': 1}
            >>> ratings_df = pd.DataFrame({'user_id': ['u1', 'u1', 'u1'],
            >>>                            'item_id': ['i1', 'i2', 'i3'],
            >>>                            'score': [4, 3, 3])
            >>> Ratings.from_dataframe(ratings_df, user_map=user_map, item_map=item_map)

        Args:
            interaction_frame: pandas DataFrame which represents the original interactions frame
            user_column: Name or positional index of the field of the DataFrame representing *users* column
            item_column: Name or positional index of the field of the DataFrame representing *items* column
            score_column: Name or positional index of the field of the DataFrame representing *score* column
            timestamp_column: Name or positional index of the field of the raw source representing *timesamp* column
            item_map: dictionary with string keys (the item ids) and integer values (the corresponding unique integer
                ids) used to create the item mapping. If not specified, it will be automatically created internally
            user_map: dictionary with string keys (the user ids) and integer values (the corresponding unique integer
                ids) used to create the user mapping. If not specified, it will be automatically created internally

        Returns:
            `Ratings` object instantiated thanks to an existing Pandas DataFrame
        """

        def get_value_row_df(row, column, dtype):
            try:
                if isinstance(column, str):
                    value = row[column]
                else:
                    # it's an int, so we get the column id and then we get the corresponding value in the row
                    key_dict = interaction_frame.columns[column]
                    value = row[key_dict]
            except (KeyError, IndexError) as e:
                if isinstance(e, KeyError):
                    raise KeyError(f"Column {column} not found in interaction frame!")
                else:
                    raise IndexError(f"Column {column} not found in interaction frame!")

            return dtype(value) if value is not None else None

        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        # lists that will contain the original data temporarily
        # this is so that the conversion from string ids to integers will be called only once
        # said lists will also be used to create the mappings if not specified in the parameters
        tmp_user_id_column = []
        tmp_item_id_column = []
        tmp_score_column = []
        tmp_timestamp_column = []

        for i, row in enumerate(interaction_frame.to_dict(orient='records')):
            user_id = get_value_row_df(row, user_column, str)
            item_id = get_value_row_df(row, item_column, str)
            score = get_value_row_df(row, score_column, float)
            timestamp = get_value_row_df(row, timestamp_column, int) if timestamp_column is not None else np.nan

            tmp_user_id_column.append(user_id)
            tmp_item_id_column.append(item_id)
            tmp_score_column.append(score)
            tmp_timestamp_column.append(timestamp)

        # create the item_map from the item_id column if not specified
        if item_map is None:
            obj.item_map = StrIntMap(np.array(list(dict.fromkeys(tmp_item_id_column))))
        else:
            obj.item_map = StrIntMap(item_map)

        # create the user_map from the user_id column if not specified
        if user_map is None:
            obj.user_map = StrIntMap(np.array(list(dict.fromkeys(tmp_user_id_column))))
        else:
            obj.user_map = StrIntMap(user_map)

        tmp_user_id_column = np.array(tmp_user_id_column)

        if np.any(tmp_user_id_column == None):
            raise UserNone('User column cannot contain None values') from None

        tmp_item_id_column = np.array(tmp_item_id_column)

        if np.any(tmp_item_id_column == None):
            raise ItemNone('Item column cannot contain None values') from None

        # convert user and item ids and create the uir matrix
        obj._uir = np.array((
            obj.user_map.convert_seq_str2int(tmp_user_id_column),
            obj.item_map.convert_seq_str2int(tmp_item_id_column),
            tmp_score_column, tmp_timestamp_column
        )).T

        obj._uir[:, 2] = obj._uir[:, 2].astype(float)
        obj._uir[:, 3] = obj._uir[:, 3].astype(float)

        # create the utility dictionary user2rows
        obj._user2rows = {
            user_idx: np.where(obj._uir[:, 0] == user_idx)[0]
            for user_idx in obj.unique_user_idx_column
        }

        return obj

    @classmethod
    @handler_score_not_float
    def from_list(cls, interaction_list: Union[List[Tuple], Iterator],
                  user_map: Union[Dict[str, int], np.ndarray, StrIntMap] = None,
                  item_map: Union[Dict[str, int], np.ndarray, StrIntMap] = None) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing list of tuples or its generator

        Furthermore, it is also possible to specify the user and item mapping between original string ids and
        integer ones. However, differently from the `Ratings` class documentation, it is possible not only to specify
        them as dictionaries but also as numpy arrays or `StrIntMap` objects directly. The end result will be the same
        independently of the type, but it is suggested to check the `StrIntMap` class documentation to understand
        the differences between the three possible types

        Examples:

            >>> interactions_list = [('u1', 'i1', 5), ('u2', 'i1', 4)]
            >>> Ratings.from_list(interactions_list)

            or

            >>> user_map = {'u1': 0, 'u2': 1}
            >>> item_map = {'i1': 0}
            >>> interactions_list = [('u1', 'i1', 5), ('u2', 'i1', 4)]
            >>> Ratings.from_list(interactions_list, user_map=user_map, item_map=item_map)

        Args:
            interaction_list: List containing tuples or its generator
            item_map: dictionary with string keys (the item ids) and integer values (the corresponding unique integer
                ids) used to create the item mapping. If not specified, it will be automatically created internally
            user_map: dictionary with string keys (the user ids) and integer values (the corresponding unique integer
                ids) used to create the user mapping. If not specified, it will be automatically created internally

        Returns:
            `Ratings` object instantiated thanks to an existing interaction list
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        # lists that will contain the original data temporarily
        # this is so that the conversion from string ids to integers will be called only once
        # said lists will also be used to create the mappings if not specified in the parameters
        tmp_user_id_column = []
        tmp_item_id_column = []
        tmp_score_column = []
        tmp_timestamp_column = []

        for i, interaction in enumerate(interaction_list):

            tmp_user_id_column.append(interaction[0])
            tmp_item_id_column.append(interaction[1])
            tmp_score_column.append(interaction[2])

            if len(interaction) == 4:
                tmp_timestamp_column.append(interaction[3])
            else:
                tmp_timestamp_column.append(np.nan)

        # create the item_map from the item_id column if not specified
        if item_map is None:
            obj.item_map = StrIntMap(np.array(list(dict.fromkeys(tmp_item_id_column))))
        else:
            obj.item_map = StrIntMap(item_map)

        # create the user_map from the user_id column if not specified
        if user_map is None:
            obj.user_map = StrIntMap(np.array(list(dict.fromkeys(tmp_user_id_column))))
        else:
            obj.user_map = StrIntMap(user_map)

        tmp_user_id_column = np.array(tmp_user_id_column)

        if np.any(tmp_user_id_column == None):
            raise UserNone('User column cannot contain None values')

        tmp_item_id_column = np.array(tmp_item_id_column)

        if np.any(tmp_item_id_column == None):
            raise ItemNone('Item column cannot contain None values')

        # convert user and item ids and create the uir matrix
        obj._uir = np.array((
            obj.user_map.convert_seq_str2int(tmp_user_id_column),
            obj.item_map.convert_seq_str2int(tmp_item_id_column),
            tmp_score_column, tmp_timestamp_column
        )).T

        obj._uir[:, 2] = obj._uir[:, 2].astype(float)
        obj._uir[:, 3] = obj._uir[:, 3].astype(float)

        # create the utility dictionary user2rows
        obj._user2rows = {
            user_idx: np.where(obj._uir[:, 0] == user_idx)[0]
            for user_idx in obj.unique_user_idx_column
        }

        return obj

    @classmethod
    def from_uir(cls, uir: np.ndarray,
                 user_map: Union[Dict[str, int], np.ndarray, StrIntMap],
                 item_map: Union[Dict[str, int], np.ndarray, StrIntMap]) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing uir matrix

        The uir matrix should be a two-dimensional numpy ndarray where each row represents a user interaction.
        Each row should be in the following format:

        ```
        [0. 0. 4] or [0. 0. 4 np.nan] (without or with the timestamp)
        ```

        In the case of a different format for the rows, a ValueError exception will be raised.
        Furthermore, if the uir matrix is not of dtype np.float64, a TypeError exception will be raised.

        In this case the 'user_map' and 'item_map' parameters ***MUST*** be specified, since there is no information
        regarding the original string ids in the uir matrix

        Examples:

            >>> uir_matrix = np.array([[0, 0, 4], [1, 0, 3]])
            >>> user_map = {'u1': 0, 'u2': 1}
            >>> item_map = {'i1': 0}
            >>> Ratings.from_uir(uir_matrix, user_map=user_map, item_map=item_map)

        Args:
            uir: uir matrix which will be used to create the new `Ratings` object
            item_map: dictionary with string keys (the item ids) and integer values (the corresponding unique integer ids)
                used to create the item mapping
            user_map: dictionary with string keys (the user ids) and integer values (the corresponding unique integer
                ids) used to create the user mapping

        Returns:
            `Ratings` object instantiated thanks to an existing uir matrix
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        if uir.shape[0] > 0 and uir.shape[1] > 0:
            if uir.shape[1] < 3:
                raise ValueError('User item ratings matrix should have at least 3 rows '
                                 '(one for users, one for items and one for ratings scores)')
            elif uir.shape[1] == 3:
                uir = np.append(uir, np.full((uir.shape[0], 1), fill_value=np.nan), axis=1)

            if uir.dtype != np.float64:
                raise TypeError('User id columns and item id columns should be mapped to their respective integer')
        else:
            uir = np.array([])

        obj._uir = uir

        obj.user_map = StrIntMap(user_map)
        obj.item_map = StrIntMap(item_map)

        obj._user2rows = {
            user_idx: np.where(obj._uir[:, 0] == user_idx)[0]
            for user_idx in obj.unique_user_idx_column
        }

        return obj

    def __len__(self):
        return self._uir.shape[0]

    def __str__(self):
        return str(self.to_dataframe(ids_as_str=True))

    def __repr__(self):
        return repr(self._uir)

    def __iter__(self):
        """
        Note: iteration is done on integer ids, if you want to iterate over string ids you need to iterate over the
        'user_id_column' or 'item_id_column'
        """
        yield from iter(self._uir)

    def __eq__(self, other):

        if isinstance(other, Ratings):
            return np.array_equal(self._uir, other._uir, equal_nan=True) and \
                self.user_map == other.user_map \
                and self.item_map == other.item_map
        else:
            return False


# Aliases for the Ratings class

class Prediction(Ratings):
    """
    This class is just an alias for the `Ratings` class, it has exactly same functionalities
    """
    pass


class Rank(Ratings):
    """
    This class is just an alias for the `Ratings` class, it has exactly same functionalities
    """
    pass

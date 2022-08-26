from __future__ import annotations

import functools
import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List, Iterable, Generator

import pandas as pd

from clayrs.content_analyzer.exceptions import Handler_ScoreNotFloat
from clayrs.content_analyzer.ratings_manager.score_processor import ScoreProcessor
from clayrs.content_analyzer.raw_information_source import RawInformationSource
from clayrs.utils.const import get_progbar
from clayrs.utils.save_content import get_valid_filename


class Interaction:
    """
    Class which models an interaction between a user and an item

    Each interaction has a score (a numeric value) and an optional timestamp

    Args:
        user_id: ID of the user
        item_id: ID of the item
        score: Numeric value of the interaction
        timestamp: Optional timestamp of the interaction
    """
    __slots__ = ('_user_id', '_item_id', '_score', '_timestamp')

    def __init__(self, user_id: str, item_id: str, score: float, timestamp: str = None):
        self._user_id = user_id
        self._item_id = item_id
        self._score = score
        self._timestamp = timestamp

    @property
    def user_id(self):
        """
        Getter for the `user_id` of the interaction
        """
        return self._user_id

    @property
    def item_id(self):
        """
        Getter for the `item_id` of the interaction
        """
        return self._item_id

    @property
    def score(self):
        """
        Getter for the `score` of the interaction
        """
        return self._score

    @property
    def timestamp(self):
        """
        Getter for the `timestamp` of the interaction. Could be `None`
        """
        return self._timestamp

    def __str__(self):
        if self.timestamp is not None:
            string = f"(user_id: {self.user_id}, item_id: {self.item_id}, score: {self.score}," \
                     f" timestamp: {self.timestamp})"
        else:
            string = f"(user_id: {self.user_id}, item_id: {self.item_id}, score: {self.score})"

        return string

    def __repr__(self):
        return f"Interaction(user_id={self.user_id}, item_id={self.item_id}, score={self.score}, " \
               f"timestamp={self.timestamp})"

    def __eq__(self, other):
        if isinstance(other, Interaction):
            timestamp_equal = (self.timestamp is None and other.timestamp is None) or \
                              (self.timestamp == other.timestamp)

            return self.user_id == other.user_id and self.item_id == other.item_id and \
                   self.score == other.score and timestamp_equal
        else:
            return False


class Ratings:
    """
    Class responsible of importing an interaction frame into the framework

    **If** the source file contains users, items and ratings in this order,
    no additional parameters are needed, **otherwise** the mapping must be explicitly specified using:

    * **'user_id'** column,
    * **'item_id'** column,
    * **'score'** column

    The *score* column can also be processed: in case you would like to consider as score the sentiment of a textual
    review, or maybe normalizing all scores in $[0, 1]$ range. Check the example below for more

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

    Args:
        source: Source containing the raw interaction frame
        user_id_column: Name or positional index of the field of the raw source representing *users* column
        item_id_column: Name or positional index of the field of the raw source representing *items* column
        score_column: Name or positional index of the field of the raw source representing *score* column
        timestamp_column: Name or positional index of the field of the raw source representing *timesamp* column
        score_processor: `ScoreProcessor` object which will process the `score_column` accordingly. Useful if you want
            to perform sentiment analysis on a textual column or you want to normalize all scores in $[0, 1]$ range
    """

    def __init__(self, source: RawInformationSource,
                 user_id_column: Union[str, int] = 0,
                 item_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: ScoreProcessor = None):

        self._ratings_dict = self._import_ratings(source, user_id_column, item_id_column, score_column,
                                                  timestamp_column, score_processor)

    @property
    @functools.lru_cache(maxsize=128)
    def user_id_column(self) -> list:
        """
        Getter for the user id column. This will return the user column "as is", so it will contain duplicate users.
        Use set(user_id_column) to get all
        unique users

        Returns:
            Users column with duplicates
        """
        return [interaction.user_id for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def item_id_column(self) -> list:
        """
        Getter for the user id column. This will return the item column "as is", so it will contain duplicate items.
        Use set(item_id_column) to get all unique users

        Returns:
            Items column with duplicates
        """
        return [interaction.item_id for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def score_column(self) -> list:
        """
        Getter for the score column. This will return the score column "as is".

        Returns:
            Score column
        """
        return [interaction.score for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def timestamp_column(self) -> list:
        """
        Getter for the timestamp column. This will return the score column "as is". If no timestamp is present then an
        empty list is returned

        Returns:
            Timestamp column or empty list if no timestamp is present
        """
        return [interaction.timestamp for interaction in self if interaction.timestamp is not None]

    @Handler_ScoreNotFloat
    def _import_ratings(self, source: RawInformationSource,
                        user_column: Union[str, int],
                        item_column: Union[str, int],
                        score_column: Union[str, int],
                        timestamp_column: Union[str, int],
                        score_processor: ScoreProcessor):
        ratings_dict = defaultdict(list)

        with get_progbar(source) as pbar:

            pbar.set_description(desc="Importing ratings")
            for row in pbar:

                user_id = self._get_field_data(user_column, row)
                item_id = self._get_field_data(item_column, row)
                score = self._get_field_data(score_column, row)
                timestamp = None

                if score_processor is not None:
                    score = score_processor.fit(score)
                else:
                    score = float(score)

                if timestamp_column is not None:
                    timestamp = self._get_field_data(timestamp_column, row)

                ratings_dict[user_id].append(Interaction(user_id, item_id, score, timestamp))

        # re-hashing
        return dict(ratings_dict)

    def get_user_interactions(self, user_id: str, head: int = None) -> List[Interaction]:
        """
        Method which returns a list of `Interaction` objects for a single user, one for each interaction of the user.
        Then you can easily iterate and extract useful information using list comprehension

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

            >>> rating_frame.get_user_interactions('u1')
            [Interaction(user_id='u1', item_id='i1', score=4),
            Interaction(user_id='u1', item_id='i2', score=3)]

            So you could easily extract all the ratings that a user has given for example:

            >>> [interaction.score for interaction in rating_frame.get_user_interactions('u1')]
            [4, 3]

            If you only want the first $k$ interactions of the user, set `head=k`. The interactions returned are the
            first $k$ according to their order of appearance in the rating frame:

            >>> rating_frame.get_user_interactions('u1', head=1)
            [Interaction(user_id='u1', item_id='i1', score=4)]

        Args:
            user_id: User for which interactions must be extracted
            head: Integer which will cut the list of interactions of the user returned. The interactions returned are
                the first $k$ according to their order of appearance

        Returns:
            List of Interaction objects of a single user

        """
        return self._ratings_dict[user_id][:head]

    def filter_ratings(self, user_list: Iterable[str]) -> Ratings:
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

            >>> rating_frame.filter_ratings(['u1'])

            ```title="Returned Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u1      | i2      |     3 |
            +---------+---------+-------+
            ```

        Args:
            user_list: List of user ids that will be present in the filtered `Ratings` object

        Returns
            The filtered Ratings object which contains only interactions of selected users
        """
        filtered_ratings_generator = ((user, self._ratings_dict[user]) for user in user_list)

        return self.from_dict(filtered_ratings_generator)

    def take_head_all(self, head: int) -> Ratings:
        """
        Method which will retain only $k$ interactions for each user. The $k$ interactions retained are the first which
        appears in the rating frame.

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

            >>> rating_frame.take_head_all(head=1)

            ```title="Returned Rating object"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u2      | i5      |     1 |
            +---------+---------+-------+
            ```

        Args:
            head: The number of interactions to retain for each user

        Returns:
            The filtered Ratings object which contains only first $k$ interactions for each user
        """

        ratings_cut_generator = ((user_id, user_ratings[:head])
                                 for user_id, user_ratings in
                                 zip(self._ratings_dict.keys(), self._ratings_dict.values()))

        return self.from_dict(ratings_cut_generator)

    # @Handler_ScoreNotFloat
    # def add_score_column(self, score_column: Union[str, int], column_name: str,
    #                      score_processor: ScoreProcessor = None):
    #
    #     col_to_add = [self._get_field_data(score_column, row) for row in self._source]
    #
    #     if score_processor:
    #         col_to_add = score_processor.fit(col_to_add)
    #     else:
    #         col_to_add = [float(score) for score in col_to_add]
    #
    #     self._ratings_frame[column_name] = col_to_add
    #
    #     start_ratings_user = 0
    #     for user_id, user_ratings in zip(self._ratings_dict.keys(), self._ratings_dict.values()):
    #         first_range_val = start_ratings_user
    #         second_range_val = start_ratings_user + len(user_ratings)
    #
    #         score_to_add = col_to_add[first_range_val:second_range_val]
    #         new_ratings = [rating_tuple + (added_score,) for rating_tuple, added_score in
    #                        zip(user_ratings, score_to_add)]
    #         self._ratings_dict[user_id] = new_ratings
    #
    #         start_ratings_user += len(user_ratings)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Method which will convert the `Rating` object to a `pandas DataFrame object`.

        The returned DataFrame object will contain the 'user_id', 'item_id' and 'score' column and optionally the
        'timestamp' column, if at least one interaction has a timestamp.

        Returns:
            The rating frame converted to a pandas DataFrame with 'user_id', 'item_id', 'score' column and optionally
            the 'timestamp' column

        """
        will_be_frame = {'user_id': self.user_id_column,
                         'item_id': self.item_id_column,
                         'score': self.score_column}

        if len(self.timestamp_column) != 0:
            will_be_frame['timestamp'] = self.timestamp_column

        return pd.DataFrame(will_be_frame)

    def to_csv(self, output_directory: str = '.', file_name: str = 'ratings_frame', overwrite: bool = False):
        """
        Method which will save the `Ratings` object to a `csv` file

        Args:
            output_directory: directory which will contain the csv file
            file_name: Name of the csv_file
            overwrite: If set to True and a csv file exists in the same output directory with the same file name, it
                will be overwritten
        """
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        file_name = get_valid_filename(output_directory, file_name, 'csv', overwrite)

        frame = self.to_dataframe()
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
    def from_dataframe(cls, interaction_frame: pd.DataFrame,
                       user_column: Union[str, int] = 0,
                       item_column: Union[str, int] = 1,
                       score_column: Union[str, int] = 2,
                       timestamp_column: Union[str, int] = None) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing pandas DataFrame

        **If** the pandas DataFrame contains users, items and ratings in this order,
        no additional parameters are needed, **otherwise** the mapping must be explicitly specified using:

        * **'user_id'** column,
        * **'item_id'** column,
        * **'score'** column

        Check documentation of the `Ratings` class for examples on mapping columns explicitly, the functioning is the
        same

        Examples:

            >>> ratings_df = pd.DataFrame({'user_id': ['u1', 'u1', 'u1'],
            >>>                            'item_id': ['i1', 'i2', 'i3'],
            >>>                            'score': [4, 3, 3])
            >>> Ratings.from_dataframe(ratings_df)

        Args:
            interaction_frame: pandas DataFrame which represents the original interactions frame
            user_column: Name or positional index of the field of the DataFrame representing *users* column
            item_column: Name or positional index of the field of the DataFrame representing *items* column
            score_column: Name or positional index of the field of the DataFrame representing *score* column
            timestamp_column: Name or positional index of the field of the raw source representing *timesamp* column

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

        ratings_dict = defaultdict(list)

        if not interaction_frame.empty:
            for row in interaction_frame.to_dict(orient='records'):
                user_id = get_value_row_df(row, user_column, str)
                item_id = get_value_row_df(row, item_column, str)
                score = get_value_row_df(row, score_column, float)
                timestamp = get_value_row_df(row, timestamp_column, str) if timestamp_column is not None else None

                ratings_dict[user_id].append(Interaction(user_id, item_id, score, timestamp))

        obj._ratings_dict = dict(ratings_dict)
        return obj

    @classmethod
    def from_list(cls, interaction_list: Union[List[Interaction], Generator]) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing list containing `Interaction`
        objects or its generator

        Examples:

            >>> interactions_list = [Interaction('u1', 'i1', 5), Interaction('u2', 'i1', 4)]
            >>> Ratings.from_list(interactions_list)

        Args:
            interaction_list: List containing `Interaction` objects or its generator

        Returns:
            `Ratings` object instantiated thanks to an existing interaction list
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        ratings_dict = defaultdict(list)
        for interaction in interaction_list:
            ratings_dict[interaction.user_id].append(interaction)

        obj._ratings_dict = dict(ratings_dict)
        return obj

    @classmethod
    def from_dict(cls, interaction_dict: Union[Dict[str, List[Interaction]], Generator]) -> Ratings:
        """
        Class method which allows to instantiate a `Ratings` object by using an existing dictionary containing
        user_id as keys and lists of `Interaction` objects as value

        Examples:

            >>> interactions_dict = {'u1': [Interaction('u1', 'i2', 4), Interaction('u1', 'i3', 3)],
            >>>                      'u2': [Interaction('u2', 'i2', 5)]}
            >>> Ratings.from_dict(interactions_dict)

        Args:
            interaction_dict: Dictionary containing user_id as keys and lists of `Interaction` objets as values
                or its generator

        Returns:
            `Ratings` object instantiated thanks to an existing dictionary
        """
        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        obj._ratings_dict = dict(interaction_dict)
        return obj

    def __len__(self):
        # all columns have same length, so only one is needed in order
        # to check what is the length
        return len(self.user_id_column)

    def __str__(self):
        return str(self.to_dataframe())

    def __repr__(self):
        return repr(self._ratings_dict)

    def __iter__(self):
        """
        The `Ratings` object can be iterated over and each iteration will return an `Interaction` object

        Examples:

            ```title="Rating object to iterate"
            +---------+---------+-------+
            | user_id | item_id | score |
            +---------+---------+-------+
            | u1      | i1      |     4 |
            | u2      | i5      |     1 |
            +---------+---------+-------+
            ```

            >>> # for simplicity we stop after the first iteration
            >>> for interaction in ratings:
            >>>     first_interaction = interaction
            >>>     break
            >>> first_interaction
            Interaction('u1', 'i1', 4)
        """
        yield from itertools.chain.from_iterable(self._ratings_dict.values())


class RatingsLowMemory:

    def __init__(self, source: RawInformationSource,
                 user_id_column: Union[str, int] = 0,
                 item_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: ScoreProcessor = None):

        rat = pd.DataFrame(source, dtype=str)
        self._ratings_dict = self._import_ratings(rat, user_id_column, item_id_column, score_column,
                                                  timestamp_column, score_processor)

    @property
    @functools.lru_cache(maxsize=128)
    def user_id_column(self) -> list:
        return self._ratings_dict.index.get_level_values('user_id').tolist()

    @property
    @functools.lru_cache(maxsize=128)
    def item_id_column(self) -> list:
        return self._ratings_dict.index.get_level_values('item_id').tolist()

    @property
    @functools.lru_cache(maxsize=128)
    def score_column(self) -> list:
        return self._ratings_dict['score'].tolist()

    @property
    @functools.lru_cache(maxsize=128)
    def timestamp_column(self) -> list:
        timestamp_list = self._ratings_dict['timestamp'].tolist()
        return timestamp_list if all(timestamp is not None for timestamp in timestamp_list) else []

    @Handler_ScoreNotFloat
    def _import_ratings(self, rat: pd.DataFrame,
                        user_column: Union[str, int],
                        item_column: Union[str, int],
                        score_column: Union[str, int],
                        timestamp_column: Union[str, int],
                        score_processor: ScoreProcessor):
        """
        Imports the ratings from the source and stores in a dataframe

        Returns:
            ratings_frame: pd.DataFrame
        """
        if isinstance(user_column, int):
            user_column = rat.columns[user_column]
        if isinstance(item_column, int):
            item_column = rat.columns[item_column]
        if isinstance(score_column, int):
            score_column = rat.columns[score_column]
        if isinstance(timestamp_column, int):
            timestamp_column = rat.columns[timestamp_column]
        elif timestamp_column is None:
            rat['timestamp'] = None
            timestamp_column = 'timestamp'

        index = pd.MultiIndex.from_tuples(zip(rat[user_column].values, rat[item_column].values),
                                          names=["user_id", "item_id"])

        rat = rat[[score_column, timestamp_column]].set_index(index)

        rat.columns = ['score', 'timestamp']

        rat['score'] = pd.to_numeric(rat['score'])

        return rat

    def get_user_interactions(self, user_id: str, head: int = None):
        user_rat = self._ratings_dict.loc[user_id][:head]

        user_rat = [Interaction(user_id, index_item, row[0], row[1])
                    for index_item, row in zip(user_rat.index, user_rat.values)]

        return user_rat

    def filter_ratings(self, user_list: Iterable[str]):
        filtered_df = self._ratings_dict.loc[
            (self._ratings_dict.index.get_level_values('user_id').isin(set(user_list)))]

        filtered_df = filtered_df.reset_index(drop=False)

        return self.from_dataframe(filtered_df, user_column='user_id',
                                   item_column='item_id',
                                   score_column='score',
                                   timestamp_column='timestamp')

    def take_head_all(self, head: int):

        filtered_df = self._ratings_dict.groupby(level='user_id').head(head)

        filtered_df = filtered_df.reset_index(drop=False)

        return self.from_dataframe(filtered_df, user_column='user_id',
                                   item_column='item_id',
                                   score_column='score',
                                   timestamp_column='timestamp')

    @classmethod
    def from_dataframe(cls, interaction_frame: pd.DataFrame,
                       user_column: Union[str, int] = 0,
                       item_column: Union[str, int] = 1,
                       score_column: Union[str, int] = 2,
                       timestamp_column: Union[str, int] = None):

        obj = cls.__new__(cls)  # Does not call __init__
        super(RatingsLowMemory, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        ratings_dict = cls._import_ratings(obj, interaction_frame, user_column, item_column, score_column,
                                           timestamp_column, None)

        obj._ratings_dict = ratings_dict
        return obj

    @classmethod
    def from_list(cls, interaction_list: List[Interaction]):

        obj = cls.__new__(cls)  # Does not call __init__
        super(RatingsLowMemory, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        user_column_iterator = (interaction.user_id for interaction in interaction_list)
        item_column_iterator = (interaction.item_id for interaction in interaction_list)

        index = pd.MultiIndex.from_tuples(zip(user_column_iterator, item_column_iterator),
                                          names=["user_id", "item_id"])

        score_timestamp_iterator = ({'score': interaction.score, 'timestamp': interaction.timestamp}
                                    for interaction in interaction_list)

        rat = pd.DataFrame(score_timestamp_iterator, index=index)

        obj._ratings_dict = rat
        return obj

    @classmethod
    def from_dict(cls, interaction_dict: Dict[str, List[Interaction]]):
        obj = cls.__new__(cls)  # Does not call __init__
        super(RatingsLowMemory, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        obj._ratings_dict = dict(interaction_dict)
        return obj

    def __iter__(self):
        yield from itertools.chain.from_iterable(self.get_user_interactions(user_id)
                                                 for user_id in
                                                 self._ratings_dict.index.get_level_values('user_id').unique())

    def __len__(self):
        return len(self._ratings_dict)


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

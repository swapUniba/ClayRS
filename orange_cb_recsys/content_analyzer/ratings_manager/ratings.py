import functools
import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List, Iterable

import pandas as pd

from orange_cb_recsys.content_analyzer.exceptions import Handler_ScoreNotFloat
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.const import get_progbar
from orange_cb_recsys.utils.save_content import get_valid_filename


class Interaction:
    __slots__ = ('_user_id', '_item_id', '_score', '_timestamp')

    def __init__(self, user_id: str, item_id: str, score: float, timestamp: str = None):
        self._user_id = user_id
        self._item_id = item_id
        self._score = score
        self._timestamp = timestamp

    @property
    def user_id(self):
        return self._user_id

    @property
    def item_id(self):
        return self._item_id

    @property
    def score(self):
        return self._score

    @property
    def timestamp(self):
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
            return self.user_id == other.user_id and self.item_id == other.item_id and \
                   self.score == other.score and self.timestamp == other.timestamp
        else:
            return False


class Ratings:
    """
    Class that imports the ratings

    Args:
        source (RawInformationSource): Source from which the ratings will be imported
        rating_configs (list<RatingsFieldConfig>):
        from_id_column (str): Name of the field containing the reference to the person who gave
            the rating (for example, the user id)
        to_id_column (str): Name of the field containing the reference to the item that a person
            rated
        timestamp_column (str): Name of the field containing the timestamp
        output_directory (str): Name of the directory where the acquired ratings will be stored
        score_combiner (str): Metric to use to combine the scores
    """

    def __init__(self, source: RawInformationSource,
                 user_id_column: Union[str, int] = 0,
                 item_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: RatingProcessor = None):

        self._ratings_dict = self._import_ratings(source, user_id_column, item_id_column, score_column,
                                                  timestamp_column, score_processor)

    @property
    @functools.lru_cache(maxsize=128)
    def user_id_column(self) -> list:
        return [interaction.user_id for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def item_id_column(self) -> list:
        return [interaction.item_id for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def score_column(self) -> list:
        return [interaction.score for interaction in self]

    @property
    @functools.lru_cache(maxsize=128)
    def timestamp_column(self) -> list:
        return [interaction.timestamp for interaction in self if interaction.timestamp is not None]

    @Handler_ScoreNotFloat
    def _import_ratings(self, source: RawInformationSource,
                        user_column: Union[str, int],
                        item_column: Union[str, int],
                        score_column: Union[str, int],
                        timestamp_column: Union[str, int],
                        score_processor: RatingProcessor):
        """
        Imports the ratings from the source and stores in a dataframe

        Returns:
            ratings_frame: pd.DataFrame
        """
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

    def get_user_interactions(self, user_id: str, head: int = None):
        return self._ratings_dict[user_id][:head]

    def filter_ratings(self, user_list: Iterable[str]):
        filtered_ratings_dict = {user: self._ratings_dict[user] for user in user_list}

        return self.from_dict(filtered_ratings_dict)

    def take_head_all(self, head: int):

        ratings_dict_cut = {user_id: user_ratings[:head]
                            for user_id, user_ratings in zip(self._ratings_dict.keys(), self._ratings_dict.values())}

        return self.from_dict(ratings_dict_cut)

    # @Handler_ScoreNotFloat
    # def add_score_column(self, score_column: Union[str, int], column_name: str,
    #                      score_processor: RatingProcessor = None):
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

    def to_dataframe(self):
        will_be_frame = {'user_id': self.user_id_column,
                         'item_id': self.item_id_column,
                         'score': self.score_column}

        if len(self.timestamp_column) != 0:
            will_be_frame['timestamp'] = self.timestamp_column

        return pd.DataFrame(will_be_frame)

    def to_csv(self, output_directory: str = '.', file_name: str = 'ratings_frame',
               overwrite: bool = False):
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
                       timestamp_column: Union[str, int] = None):

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
    def from_list(cls, interaction_list: List[Interaction]):

        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        ratings_dict = defaultdict(list)
        for interaction in interaction_list:
            ratings_dict[interaction.user_id].append(interaction)

        obj._ratings_dict = dict(ratings_dict)
        return obj

    @classmethod
    def from_dict(cls, interaction_dict: Dict[str, List[Interaction]]):
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
        yield from itertools.chain.from_iterable(self._ratings_dict.values())


class RatingsLowMemory:

    def __init__(self, source: RawInformationSource,
                 user_id_column: Union[str, int] = 0,
                 item_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: RatingProcessor = None):

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
                        score_processor: RatingProcessor):
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
        filtered_df = self._ratings_dict.loc[(self._ratings_dict.index.get_level_values('user_id').isin(set(user_list)))]

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
                                                 for user_id in self._ratings_dict.index.get_level_values('user_id').unique())

    def __len__(self):
        return len(self._ratings_dict)


# Aliases for the Ratings class

class Prediction(Ratings):
    pass


class Rank(Ratings):
    pass

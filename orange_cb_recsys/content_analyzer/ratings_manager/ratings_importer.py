import itertools
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List, Iterable

import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from orange_cb_recsys.content_analyzer.exceptions import Handler_ScoreNotFloat
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.const import get_pbar
from orange_cb_recsys.utils.save_content import get_valid_filename


class Interaction:
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
        return str(self)


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

        self._ratings_dict, self._user_id_column, self._item_id_column, self._score_column, self._timestamp_column = \
            self._import_ratings(source, user_id_column, item_id_column, score_column, timestamp_column, score_processor)

    @property
    def user_id_column(self) -> list:
        return self._user_id_column

    @property
    def item_id_column(self) -> list:
        return self._item_id_column

    @property
    def score_column(self) -> list:
        return self._score_column

    @property
    def timestamp_column(self) -> list:
        return self._timestamp_column

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
        ratings_user_score = defaultdict(list)
        ratings_user_item = defaultdict(list)
        ratings_user_timestamp = defaultdict(list)

        with logging_redirect_tqdm():
            pbar = get_pbar(list(source))
            pbar.set_description(desc="Importing ratings")
            for row in pbar:

                user_id = self._get_field_data(user_column, row)
                item_id = self._get_field_data(item_column, row)
                score = self._get_field_data(score_column, row)

                ratings_user_item[user_id].append(item_id)
                ratings_user_score[user_id].append(score)

                if timestamp_column is not None:
                    timestamp = self._get_field_data(timestamp_column, row)
                    ratings_user_timestamp[user_id].append(timestamp)

        scores_list = list(itertools.chain.from_iterable(ratings_user_score.values()))
        if score_processor:
            scores_list = score_processor.fit(scores_list)
        else:
            scores_list = [float(score) for score in scores_list]

        start_index = 0
        for user_id in ratings_user_score:
            n_user_ratings = len(ratings_user_score[user_id])
            second_index = start_index + n_user_ratings

            user_score_processed = scores_list[start_index:second_index]

            if timestamp_column is not None:
                ratings_dict[user_id] = [Interaction(user_id, item_id, score)
                                         for item_id, score, timestamp in zip(ratings_user_item[user_id],
                                                                              user_score_processed,
                                                                              ratings_user_timestamp[user_id])]
            else:
                ratings_dict[user_id] = [Interaction(user_id, item_id, score)
                                         for item_id, score in zip(ratings_user_item[user_id], user_score_processed)]

            start_index += n_user_ratings

        user_id_column_final = [user_id for user_id in ratings_user_item.keys() for _ in
                                range(len(ratings_user_item[user_id]))]

        item_id_column_final = list(itertools.chain.from_iterable(ratings_user_item.values()))

        score_column_final = scores_list

        timestamp_column_final = []
        if timestamp_column is not None:
            timestamp_column_final = list(itertools.chain.from_iterable(ratings_user_timestamp.values()))

        return ratings_dict, user_id_column_final, item_id_column_final, score_column_final, timestamp_column_final

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

        def get_column_df(column, dtype):
            try:
                if isinstance(column, str):
                    column_values = list(interaction_frame[column].values.astype(dtype))
                else:
                    column_values = list(interaction_frame.iloc[:, column].values.astype(dtype))
            except (KeyError, IndexError) as e:
                if isinstance(e, KeyError):
                    raise KeyError(f"Column {column} not found in interaction frame!")
                else:
                    raise IndexError(f"Column {column} not found in interaction frame!")

            return column_values

        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        ratings_dict = defaultdict(list)
        obj._user_id_column = []
        obj._item_id_column = []
        obj._score_column = []
        obj._timestamp_column = []

        if not interaction_frame.empty:
            obj._user_id_column = get_column_df(user_column, str)
            obj._item_id_column = get_column_df(item_column, str)
            obj._score_column = get_column_df(score_column, float)
            obj._timestamp_column = [None for _ in range(len(obj._user_id_column))]
            if timestamp_column is not None:
                obj._timestamp_column = get_column_df(timestamp_column, str)

            for user_id, item_id, score, timestamp in zip(obj._user_id_column, obj._item_id_column,
                                                          obj._score_column, obj._timestamp_column):
                ratings_dict[user_id].append(Interaction(user_id, item_id, score, timestamp))

            if timestamp_column is None:
                obj._timestamp_column = []

        obj._ratings_dict = dict(ratings_dict)
        return obj

    @classmethod
    def from_list(cls, interaction_list: List[Interaction]):

        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        obj._user_id_column = [interaction.user_id for interaction in interaction_list]
        obj._item_id_column = [interaction.item_id for interaction in interaction_list]
        obj._score_column = [interaction.score for interaction in interaction_list]
        obj._timestamp_column = [interaction.timestamp
                                 for interaction in interaction_list if interaction.timestamp is not None]

        ratings_dict = defaultdict(list)
        for interaction in interaction_list:
            ratings_dict[interaction.user_id].append(interaction)

        obj._ratings_dict = dict(ratings_dict)
        return obj

    @classmethod
    def from_dict(cls, interaction_dict: Dict[str, List[Interaction]]):
        obj = cls.__new__(cls)  # Does not call __init__
        super(Ratings, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        obj._user_id_column = [interaction.user_id
                               for interaction in itertools.chain.from_iterable(interaction_dict.values())]
        obj._item_id_column = [interaction.item_id
                               for interaction in itertools.chain.from_iterable(interaction_dict.values())]
        obj._score_column = [interaction.score
                             for interaction in itertools.chain.from_iterable(interaction_dict.values())]
        obj._timestamp_column = [interaction.timestamp
                                 for interaction in itertools.chain.from_iterable(interaction_dict.values())
                                 if interaction.timestamp is not None]

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


# Aliases for the Ratings class

class Prediction(Ratings):
    pass


class Rank(Ratings):
    pass

import os
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

from orange_cb_recsys.content_analyzer.exceptions import Handler_ScoreNotFloat
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.const import progbar


class RatingsImporter:
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
                 from_id_column: Union[str, int] = 0,
                 to_id_column: Union[str, int] = 1,
                 score_column: Union[str, int] = 2,
                 timestamp_column: Union[str, int] = None,
                 score_processor: RatingProcessor = None):

        self.__source = source
        self.__from_id_column = from_id_column
        self.__to_id_column = to_id_column
        self.__score_column = score_column
        self.__timestamp_column = timestamp_column
        self.__score_processor = score_processor

        self.__rating_frame: pd.DataFrame = pd.DataFrame(columns=['from_id', 'to_id', 'score'])

    @property
    def from_id_column(self) -> Union[str, int]:
        return self.__from_id_column

    @property
    def to_id_column(self) -> Union[str, int]:
        return self.__to_id_column

    @property
    def score_column(self) -> Union[str, int]:
        return self.__score_column

    @property
    def timestamp_column(self) -> Union[str, int]:
        return self.__timestamp_column

    @property
    def score_processor(self) -> RatingProcessor:
        return self.__score_processor

    @property
    def rating_frame(self) -> pd.DataFrame:
        return self.__rating_frame

    @rating_frame.setter
    def rating_frame(self, frame: pd.DataFrame):
        self.__rating_frame = frame

    @Handler_ScoreNotFloat
    def import_ratings(self) -> pd.DataFrame:
        """
        Imports the ratings from the source and stores in a dataframe

        Returns:
            ratings_frame: pd.DataFrame
        """
        ratings_frame = {'from_id': [], 'to_id': [], 'score': [], 'timestamp': []}
        for row in progbar(list(self.__source), prefix="Importing ratings:"):

            ratings_frame['from_id'].append(self._get_field_data(self.from_id_column, row))

            ratings_frame['to_id'].append(self._get_field_data(self.to_id_column, row))

            if self.timestamp_column:
                ratings_frame['timestamp'].append(self._get_field_data(self.timestamp_column, row))

            ratings_frame['score'].append(self._get_field_data(self.score_column, row))

        if len(ratings_frame['timestamp']) == 0:
            del ratings_frame['timestamp']

        if self.score_processor:
            ratings_frame['score'] = self.score_processor.fit(ratings_frame['score'])
        else:
            ratings_frame['score'] = [float(score) for score in ratings_frame['score']]

        self.rating_frame = pd.DataFrame(ratings_frame)
        return self.rating_frame

    def imported_ratings_to_csv(self, output_directory: str = '.', file_name: str = 'ratings_frame', overwrite: bool = False):
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        file_name = self._get_valid_filename(output_directory, file_name, 'csv', overwrite)
        self.rating_frame.to_csv(os.path.join(output_directory, file_name), index=False, header=True)

    @Handler_ScoreNotFloat
    def add_score_column(self, score_column: Union[str, int], column_name: str, score_processor: RatingProcessor = None):
        col_to_add = []
        for row in progbar(list(self.__source), prefix="Adding column {}:".format(column_name)):

            col_to_add.append(self._get_field_data(score_column, row))

        if score_processor:
            col_to_add = score_processor.fit(col_to_add)
        else:
            col_to_add = [float(score) for score in col_to_add]

        self.rating_frame[column_name] = col_to_add

        return self.rating_frame

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

    @staticmethod
    def _get_valid_filename(output_directory: str, filename: str, format: str, overwrite: bool):
        filename_try = "{}.{}".format(filename, format)

        if overwrite is False:
            i = 0
            while os.path.isfile(os.path.join(output_directory, filename_try)):
                i += 1
                filename_try = "{} ({}).{}".format(filename, i, format)

        return filename_try

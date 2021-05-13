from typing import List

import pandas as pd

from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.content_analyzer.ratings_manager.score_combiner import ScoreCombiner
from orange_cb_recsys.utils.const import home_path, logger, DEVELOPING


class RatingsFieldConfig:
    """
    Class for the configuration of the field containing the ratings

    Args:
        field_name (str): Name of the field that contains the ratings
        processor (RatingProcessor): Processor for the data in the rating field
    """
    def __init__(self, field_name: str,
                 processor: RatingProcessor):
        self.__field_name = field_name
        self.__processor = processor

    @property
    def field_name(self):
        return self.__field_name

    @property
    def processor(self):
        return self.__processor


class RatingsImporter:
    """
    Class that imports the ratings

    Args:
        source (RawInformationSource): Source from which the ratings will be imported
        rating_configs (list<RatingsFieldConfig>):
        from_field_name (str): Name of the field containing the reference to the person who gave
            the rating (for example, the user id)
        to_field_name (str): Name of the field containing the reference to the item that a person
            rated
        timestamp_field_name (str): Name of the field containing the timestamp
        output_directory (str): Name of the directory where the acquired ratings will be stored
        score_combiner (str): Metric to use to combine the scores
    """
    def __init__(self, source: RawInformationSource,
                 rating_configs: List[RatingsFieldConfig],
                 from_field_name: str,
                 to_field_name: str,
                 timestamp_field_name: str,
                 output_directory: str = None,
                 score_combiner: str = "avg"):

        self.__source: RawInformationSource = source
        self.__file_name: str = output_directory
        self.__rating_configs: List[RatingsFieldConfig] = rating_configs
        self.__from_field_name: str = from_field_name
        self.__to_field_name: str = to_field_name
        self.__timestamp_field_name: str = timestamp_field_name
        self.__score_combiner = ScoreCombiner(score_combiner)

        if not isinstance(self.__rating_configs, list):
            self.__rating_configs = [self.__rating_configs]

        self.__columns: list = ["from_id", "to_id", "score", "timestamp"]
        for field in self.__rating_configs:
            self.__columns.append(field.field_name)

    @property
    def frame_columns(self) -> list:
        return self.__columns

    @property
    def from_field_name(self) -> str:
        return self.__from_field_name

    @property
    def to_field_name(self) -> str:
        return self.__to_field_name

    @property
    def timestamp_field_name(self) -> str:
        return self.__timestamp_field_name

    def import_ratings(self) -> pd.DataFrame:
        """
        Imports the ratings from the source and stores in a dataframe

        Returns:
            ratings_frame: pd.DataFrame
        """
        ratings_frame = pd.DataFrame(columns=list(self.__columns))

        dicts = \
            [
                {
                    **{
                        "from_id": raw_rating[self.__from_field_name],
                        "to_id": raw_rating[self.__to_field_name],
                        "timestamp": raw_rating[self.__timestamp_field_name],
                        "score": self.__score_combiner.combine(
                            [preference.processor.fit(raw_rating[preference.field_name])
                             for preference in self.__rating_configs])
                    },
                    **{
                        preference.field_name:
                            raw_rating[preference.field_name]
                        for preference in self.__rating_configs
                    },
                    **{
                        "{}_score".format(preference.field_name.lower()):
                            preference.processor.fit(raw_rating[preference.field_name])
                        for preference in self.__rating_configs
                    }
                }
                for raw_rating in show_progress(self.__source)
            ]

        ratings_frame = ratings_frame.append(dicts, ignore_index=True)

        if self.__file_name is not None:
            if not DEVELOPING:
                ratings_frame.to_csv(
                    "{}/ratings/{}.csv".format(
                        home_path, self.__file_name),
                    index=False, header=True)
            else:
                ratings_frame.to_csv(
                    "{}.csv".format(
                        self.__file_name), index=False, header=True)

        return ratings_frame


def show_progress(coll, milestones=100):
    """
    Yields the elements contained in coll and prints to video how many have been processed

    Args:
        coll (list): List that contains the ratings to process
        milestones (int): Tells to the method how often he has to print an update. For
            example, if milestones = 100, for every 100 items processed the method will
            print an update
    """
    processed = 0
    for element in coll:
        yield element
        processed += 1
        if processed % milestones == 0:
            logger.info('Processed %s elements', processed)

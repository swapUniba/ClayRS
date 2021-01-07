from abc import ABC
from typing import Dict, List
import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content


class Algorithm(ABC):
    """
    Abstract class for the algorithms
    Args:
        item_field (str): Field on which execute the algorithm
        item_field_representation (str): Field representation to consider
        additional_item_fields (Dict<str, str>)
        additional_user_fields (Dict<str, str>)
    """
    def __init__(self, item_field: str, item_field_representation: str,
                 additional_item_fields: Dict[str, str] = None,
                 additional_user_fields: Dict[str, str] = None):
        super().__init__()
        if additional_item_fields is None:
            additional_item_fields = {}
        if additional_user_fields is None:
            additional_user_fields = {}
        self.__additional_item_fields = additional_item_fields
        self.__additional_user_fields = additional_user_fields
        self.__item_field: str = item_field
        self.__item_field_representation: str = item_field_representation

    def append_item_field(self, field: str, field_representation: str):
        self.__additional_item_fields[field] = field_representation

    def append_user_field(self, field: str, field_representation: str):
        self.__additional_user_fields[field] = field_representation

    @property
    def additional_item_fields(self):
        return self.__additional_item_fields

    @property
    def additional_user_fields(self):
        return self.__additional_user_fields

    @property
    def item_field(self):
        return self.__item_field

    @property
    def item_field_representation(self):
        return self.__item_field_representation

    @item_field.setter
    def item_field(self, item_field: str):
        self.__item_field = item_field

    @item_field_representation.setter
    def item_field_representation(self, item_field_representation: str):
        self.__item_field_representation = item_field_representation


class RankingAlgorithm(Algorithm):
    """
    Abstract class for the ranking algorithms
    """
    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        """
        Args:
            candidate_item_id_list: list of the items that can be recommended, if None
                all unrated items will be used
            user_id: user for which recommendations will be computed
            recs_number (list[Content]): How long the ranking will be
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            items_directory (str): Name of the directory where the items are stored.
        """
        raise NotImplementedError


class ScorePredictionAlgorithm(Algorithm):
    """
    Abstract class for the score prediction algorithms
    """
    def predict(self, user_id: str, items: List[Content], ratings: pd.DataFrame, items_directory: str):
        raise NotImplementedError

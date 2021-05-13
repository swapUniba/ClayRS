from typing import List

from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyNegativeItems
from orange_cb_recsys.recsys.content_based_algorithm.similarities import Similarity
from orange_cb_recsys.content_analyzer.content_representation.content_field import EmbeddingField, FeaturesBagField
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import get_unrated_items, get_rated_items, load_content_instance


class CentroidVector(ContentBasedAlgorithm):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked.
    Then computes the similarity between the centroid and the item of which the rating must be predicted.

    Args:
        item_field (dict): dict where the key is the name of the field
            that contains the content to use, value is the representation(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
            Check the example above for more.
        similarity (Similarity): Kind of similarity to use
        threshold (float): Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive
    """

    def __init__(self, item_field: dict, similarity: Similarity, threshold: float = 0):
        super().__init__(item_field, threshold)
        self.__similarity = similarity
        self.__centroid: np.array = None
        self.__positive_rated_dict: dict = None

    @staticmethod
    def __check_representation(representation, representation_name: str, item_field: str):
        """
        Checks that the passed representation is an embedding (in which case the granularity must be document)
        or a tf-idf vector, otherwise throws an exception because in these scenarios the centroid calculation
        cannot be computed

        Args:
            representation: representation instance
            representation_name (str): name of the item representation
            item_field (str): name of the field that has said representation
        """
        if not isinstance(representation, EmbeddingField) and \
                not isinstance(representation, FeaturesBagField):
            raise ValueError(
                "The representation %s for the %s field is not an embedding or a tf-idf vector"
                % (representation_name, item_field))

        if isinstance(representation, EmbeddingField):
            if len(representation.value.shape) != 1:
                raise ValueError(
                    "The representation %s for the %s field is not a document embedding, "
                    "so the centroid cannot be calculated" % (representation_name, item_field))

    def process_rated(self, user_ratings: pd.DataFrame, items_directory: str):
        """
        Function that extracts features from positive rated items ONLY!
        The extracted features will be used to fit the algorithm (build the query).

        Features extracted will be stored in private attributes of the class.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        # Load rated items from the path
        rated_items = get_rated_items(items_directory, user_ratings)

        # Calculates labels and extract features from the positive rated items
        positive_rated_dict = {}

        for item in rated_items:
            score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)
            if item is not None and score_assigned >= self.threshold:

                positive_rated_dict[item] = self.extract_features_item(item)

                for field in self.item_field.keys():
                    for repr_id in self.item_field[field]:
                        representation = item.get_field(field).get_representation(repr_id)
                        self.__check_representation(
                            representation, repr_id, field)

        if len(user_ratings) == 0:
            raise NoRatedItems("No rated items available locally!\n"
                               "The score frame will be empty for the user")
        if len(positive_rated_dict) == 0:
            raise OnlyNegativeItems("There are only negative items available locally!\n"
                                    "The score frame will be empty for the user")

        self.__positive_rated_dict = positive_rated_dict

    def fit(self):
        """
        The fit process for the Centroid Vector consists in calculating the centroid of the features
        of the positive items ONLY.

        This method uses extracted features of the positive items stored in a private attribute, so
        process_rated() must be called before this method.

        The built centroid will also be stored in a private attribute.
        """
        positive_rated_features = list(self.__positive_rated_dict.values())

        positive_rated_features_fused = self.fuse_representations(positive_rated_features)
        self.__centroid = np.array(positive_rated_features_fused).mean(axis=0)

    def predict(self, user_ratings: pd.DataFrame, items_directory: str,
                filter_list: List[str] = None) -> pd.DataFrame:
        """
        Predicts how much a user will like unrated items.

        One can specify which items must be predicted with the filter_list parameter,
        in this case ONLY items in the filter_list will be predicted.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be predicted.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
        """

        # Load items to predict
        if filter_list is None:
            items_to_predict = get_unrated_items(items_directory, user_ratings)
        else:
            items_to_predict = [load_content_instance(items_directory, item_id) for item_id in filter_list]

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            id_items_to_predict.append(item.content_id)
            features_items_to_predict.append(self.extract_features_item(item))

        # Calculate predictions
        logger.info("Computing similarity between centroid and unrated items")
        features_fused = self.fuse_representations(features_items_to_predict)
        similarities = [self.__similarity.perform(self.__centroid, item) for item in features_fused]

        # Build the score frame
        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        for item_id, similarity in zip(id_items_to_predict, similarities):
            score_frame = pd.concat(
                [score_frame,
                 pd.DataFrame.from_records([(item_id, similarity)], columns=columns)],
                ignore_index=True)

        return score_frame

    def rank(self, user_ratings: pd.DataFrame, items_directory: str, recs_number: int = None,
             filter_list: List[str] = None) -> pd.DataFrame:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All items will be ranked.

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list will be used to calculate the rank.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be used to calculate the rank.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
        """

        result = self.predict(user_ratings, items_directory, filter_list)

        result.sort_values(by=['rating'], ascending=False, inplace=True)

        rank = result.head(recs_number)

        return rank

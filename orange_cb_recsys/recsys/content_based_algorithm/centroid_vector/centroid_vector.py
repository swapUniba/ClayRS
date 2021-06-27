from typing import List

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyNegativeItems, NotPredictionAlg
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import Similarity
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

    def __init__(self, item_field: dict, similarity: Similarity, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)
        self.__similarity = similarity
        self.__embedding_combiner = embedding_combiner
        self.__centroid: np.array = None
        self.__positive_rated_dict: dict = None

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

        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        for item in rated_items:
            score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)
            if item is not None and score_assigned >= threshold:

                positive_rated_dict[item] = self.extract_features_item(item)

        if len(rated_items) == 0 or all(rated_items) is None:
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
        self._set_transformer()

        positive_rated_features = list(self.__positive_rated_dict.values())

        positive_rated_features_fused = self.fuse_representations(positive_rated_features, self.__embedding_combiner)
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
        raise NotPredictionAlg("CentroidVector is not a Score Prediction Algorithm!")

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
        # Load items to predict
        if filter_list is None:
            items_to_predict = get_unrated_items(items_directory, user_ratings)
        else:
            items_to_predict = [load_content_instance(items_directory, item_id) for item_id in filter_list]

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Calculate predictions
            logger.info("Computing similarity between centroid and unrated items")
            features_fused = self.fuse_representations(features_items_to_predict, self.__embedding_combiner)
            similarities = [self.__similarity.perform(self.__centroid, item) for item in features_fused]
        else:
            similarities = []

        # Build the score frame
        result = {'to_id': id_items_to_predict, 'score': similarities}

        result = pd.DataFrame(result, columns=['to_id', 'score'])

        # Sort them in descending order
        result.sort_values(by=['score'], ascending=False, inplace=True)

        rank = result[:recs_number]

        return rank

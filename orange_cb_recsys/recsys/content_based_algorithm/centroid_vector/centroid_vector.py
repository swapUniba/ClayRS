from typing import List

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyNegativeItems, \
    NotPredictionAlg, EmptyUserRatings
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import Similarity
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import recsys_logger
from orange_cb_recsys.utils.load_content import get_unrated_items, get_rated_items, \
    get_chosen_items


class CentroidVector(ContentBasedAlgorithm):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked.
    Then computes the similarity between the centroid and the item of which the ranking score must be predicted.
    It's a ranking algorithm, so it can't do score prediction

    USAGE:
        > # Interested in only a field representation, CosineSimilarity as similarity,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = CentroidVector({"Plot": 0}, CosineSimilarity(), 3)

        > # Interested in multiple field representations of the items, CosineSimilarity as similarity,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = CentroidVector(
        >                      item_field={"Plot": [0, "tfidf"],
        >                                  "Genre": [0, 1],
        >                                  "Director": "doc2vec"},
        >                      similarity=CosineSimilarity(),
        >                      threshold=3)

        > # After instantiating the CentroidVector algorithm, pass it in the initialization of
        > # a CBRS and the use its method to calculate ranking for single user or multiple users:
        > cbrs = ContentBasedRS(algorithm=alg, ...)
        > cbrs.fit_rank(...)
        > ...
        > # Check the corresponding method documentation for more

    Args:
        item_field (dict): dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
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
        The extracted features will be used to fit the algorithm (build the centroid).

        Features extracted will be stored in a private attributes of the class.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        # Load rated items from the path
        rated_items = get_rated_items(items_directory, user_ratings)

        recsys_logger.info("Processing rated items")
        # If threshold wasn't passed in the constructor, then we take the mean rating
        # given by the user as its threshold
        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Calculates labels and extract features from the positive rated items
        positive_rated_dict = {}
        for item in rated_items:
            score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)
            if item is not None and score_assigned >= threshold:

                positive_rated_dict[item] = self.extract_features_item(item)

        if user_ratings.empty:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings.from_id.iloc[0]
        if len(rated_items) == 0 or all(rated_items) is None:
            raise NoRatedItems("User {} - No rated items available locally!".format(user_id))
        if len(positive_rated_dict) == 0:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!")

        self.__positive_rated_dict = positive_rated_dict

    def fit(self):
        """
        The fit process for the CentroidVector consists in calculating the centroid of the features
        of the positive items ONLY.

        This method uses extracted features of the positive items stored in a private attribute, so
        process_rated() must be called before this method.

        The built centroid will also be stored in a private attribute.
        """
        recsys_logger.info("Calculating centroid vector")
        self._set_transformer()

        positive_rated_features = list(self.__positive_rated_dict.values())

        positive_rated_features_fused = self.fuse_representations(positive_rated_features, self.__embedding_combiner)
        self.__centroid = np.array(positive_rated_features_fused).mean(axis=0)

    def predict(self, user_ratings: pd.DataFrame, items_directory: str,
                filter_list: List[str] = None) -> pd.DataFrame:
        """
        CentroidVector is not a score prediction algorithm, calling this method will raise
        the 'NotPredictionAlg' exception!
        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("CentroidVector is not a Score Prediction Algorithm!")

    def rank(self, user_ratings: pd.DataFrame, items_directory: str, recs_number: int = None,
             filter_list: List[str] = None) -> pd.DataFrame:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list parameter will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be ranked.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            recs_number (int): number of the top items that will be present in the ranking, if None
                all unrated items will be ranked
            filter_list (list): list of the items to rank, if None all unrated items will be ranked
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
        """
        # Load items to predict
        if filter_list is None:
            items_to_predict = get_unrated_items(items_directory, user_ratings)
        else:
            items_to_predict = get_chosen_items(items_directory, filter_list)

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        recsys_logger.info("Calculating rank")
        if len(id_items_to_predict) > 0:
            # Calculate predictions, they are the similarity of the new items with the centroid vector
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

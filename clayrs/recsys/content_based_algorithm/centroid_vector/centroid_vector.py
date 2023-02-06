from __future__ import annotations
from collections import defaultdict
from typing import List, Union, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import Content
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
    from clayrs.recsys.content_based_algorithm.centroid_vector.similarities import Similarity
    from clayrs.content_analyzer import Ratings

from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from clayrs.recsys.content_based_algorithm.content_based_algorithm import PerUserCBAlgorithm
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyNegativeItems, \
    NotPredictionAlg, EmptyUserRatings


class CentroidVector(PerUserCBAlgorithm):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked.
    Then computes the similarity between the centroid and the item of which the ranking score must be predicted.
    It's a ranking algorithm, so it can't do score prediction

    *   It computes the centroid vector of the features of items *liked by the user*
    *   It computes the similarity between the centroid vector and the items of which the ranking score must be
    predicted

    The items liked by a user are those having a rating higher or equal than a specific **threshold**.
    If the threshold is not specified, the average score of all items liked by the user is used.

    Examples:
        * Interested in only a field representation, CosineSimilarity as similarity,
        $threshold = 3$ (Every item with rating $>= 3$ will be considered as positive)

        >>> from clayrs import recsys as rs
        >>> alg = rs.CentroidVector({"Plot": 0}, rs.CosineSimilarity(), 3)

        * Interested in multiple field representations of the items, CosineSimilarity as similarity,
        $threshold = None$ (Every item with rating $>=$ mean rating of the user will be considered as positive)

        >>> alg = rs.CentroidVector(
        >>>                      item_field={"Plot": [0, "tfidf"],
        >>>                                  "Genre": [0, 1],
        >>>                                  "Director": "doc2vec"},
        >>>                      similarity=rs.CosineSimilarity(),
        >>>                      threshold=None)

        !!! info

            After instantiating the CentroidVector algorithm, pass it in the initialization of
            a CBRS and the use its method to calculate ranking for single user or multiple users:

            Examples:
                 
                >>> cbrs = rs.ContentBasedRS(algorithm=alg, ...)
                >>> cbrs.fit_rank(...)
                >>> # ...

    Args:
        item_field: dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
        similarity: Kind of similarity to use
        threshold: Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive. If the threshold is not specified, the average score of all items liked by the user is used.
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used but they are in a
            matrix form instead of a single vector (e.g. when WordEmbedding representations must be used you have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_similarity', '_emb_combiner', '_centroid', '_positive_rated_list')

    def __init__(self, item_field: dict, similarity: Similarity, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)

        self._similarity = similarity
        self._emb_combiner = embedding_combiner
        self._centroid: Optional[np.ndarray] = None
        self._positive_rated_list: Optional[List] = None

    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from positive rated items ONLY!
        The extracted features will be used to fit the algorithm (build the centroid).

        Features extracted will be stored in a private attributes of the class.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
        """
        # a list since there could be duplicate interaction (eg bootstrap partitioning)
        uir_user = train_ratings.get_user_interactions(user_idx)
        rated_items_id = train_ratings.item_map.convert_seq_int2str(uir_user[:, 1].astype(int))

        # a list since there could be duplicate interaction (eg bootstrap partitioning)
        items_scores_dict = defaultdict(list)

        for item_id, score in zip(rated_items_id, uir_user[:, 2]):
            items_scores_dict[item_id].append(score)

        items_scores_dict = dict(sorted(items_scores_dict.items()))  # sort dictionary based on key for reproducibility

        # Create list of all the available items that are useful for the user
        loaded_rated_items: List[Union[Content, None]] = available_loaded_items.get_list([item_id
                                                                                          for item_id
                                                                                          in rated_items_id])

        # If threshold wasn't passed in the constructor, then we take the mean rating
        # given by the user as its threshold
        threshold = self.threshold
        if threshold is None:
            threshold = np.nanmean(uir_user[:, 2])

        # we extract feature of each POSITIVE item sorted based on its key: IMPORTANT for reproducibility!!
        # otherwise the matrix we feed to sklearn will have input item in different rows each run!
        positive_rated_list = []
        for item in loaded_rated_items:
            if item is not None:

                score_assigned = items_scores_dict[item.content_id]

                for score in score_assigned:
                    if score >= threshold:
                        positive_rated_list.append(self.extract_features_item(item))

        if len(uir_user) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        if len(loaded_rated_items) == 0 or (loaded_rated_items.count(None) == len(loaded_rated_items)):
            raise NoRatedItems("User {} - No rated items available locally!".format(user_idx))
        if len(positive_rated_list) == 0:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!")

        self._positive_rated_list = positive_rated_list

    def fit_single_user(self):
        """
        The fit process for the CentroidVector consists in calculating the centroid of the features
        of the positive items ONLY.

        This method uses extracted features of the positive items stored in a private attribute, so
        `process_rated()` must be called before this method.

        The built centroid will also be stored in a private attribute.
        """
        positive_rated_features_fused = self.fuse_representations(self._positive_rated_list, self._emb_combiner)
        # reshape make the centroid bidimensional of shape (1, h) needed to compute faster similarities
        self._centroid = positive_rated_features_fused.mean(axis=0).reshape(1, -1)

        # we delete variable used to fit since will no longer be used
        self._positive_rated_list = None

    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                            filter_list: List[str]) -> np.ndarray:
        """
        CentroidVector is not a score prediction algorithm, calling this method will raise
        the `NotPredictionAlg` exception!

        Raises:
            NotPredictionAlg: exception raised since the CentroidVector algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("CentroidVector is not a Score Prediction Algorithm!")

    def rank_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                         recs_number, filter_list: List[str]) -> np.ndarray:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked with the `filter_list` parameter,
        in this case ONLY items in the `filter_list` parameter will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, **ALL** unrated items will be ranked.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            filter_list (list): list of the items to rank, if None all unrated items for the user will be ranked

        Returns:
            List of Interactions object in a descending order w.r.t the 'score' attribute, representing the ranking for
                a single user
        """
        uir_user = train_ratings.get_user_interactions(user_idx)
        if len(uir_user) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        # Load items to predict
        items_to_predict = available_loaded_items.get_list(filter_list)

        # Extract features of the items to predict
        idx_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                idx_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(idx_items_to_predict) == 0:
            return np.array([])  # if no item to predict, empty rank is returned

        idx_items_to_predict = train_ratings.item_map.convert_seq_str2int(idx_items_to_predict)

        # Calculate predictions, they are the similarity of the new items with the centroid vector
        features_fused = self.fuse_representations(features_items_to_predict, self._emb_combiner)
        similarities = self._similarity.perform(self._centroid, features_fused).reshape(-1)  # 2d to 1d

        sorted_scores_idxs = np.argsort(similarities)[::-1][:recs_number]
        sorted_items = np.array(idx_items_to_predict)[sorted_scores_idxs]
        sorted_scores = similarities[sorted_scores_idxs]

        # we construct the output data
        uir_rank = np.array([[user_idx, item_idx, score]
                             for item_idx, score in zip(sorted_items, sorted_scores)])

        return uir_rank

    def __str__(self):
        return "CentroidVector"

    def __repr__(self):
        return f'CentroidVector(item_field={self.item_field}, ' \
               f'similarity={self._similarity}, ' \
               f'threshold={self.threshold}, ' \
               f'embedding_combiner={self._emb_combiner})'

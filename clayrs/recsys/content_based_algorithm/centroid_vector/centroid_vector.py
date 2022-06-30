from typing import List, Union, Optional, Dict

from clayrs.content_analyzer.content_representation.content import Content
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyNegativeItems, \
    NotPredictionAlg, EmptyUserRatings
from clayrs.recsys.content_based_algorithm.centroid_vector.similarities import Similarity
import numpy as np


class CentroidVector(ContentBasedAlgorithm):
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
            vector for each word). By default the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_similarity', '_emb_combiner', '_centroid', '_positive_rated_dict')

    def __init__(self, item_field: dict, similarity: Similarity, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)

        self._similarity = similarity
        self._emb_combiner = embedding_combiner
        self._centroid: Optional[np.ndarray] = None
        self._positive_rated_dict: Optional[Dict] = None

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from positive rated items ONLY!
        The extracted features will be used to fit the algorithm (build the centroid).

        Features extracted will be stored in a private attributes of the class.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
        """
        items_scores_dict = {interaction.item_id: interaction.score for interaction in user_ratings}

        # Load rated items from the path
        loaded_rated_items: List[Union[Content, None]] = available_loaded_items.get_list([item_id
                                                                                          for item_id in
                                                                                          items_scores_dict.keys()])

        # If threshold wasn't passed in the constructor, then we take the mean rating
        # given by the user as its threshold
        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Calculates labels and extract features from the positive rated items
        positive_rated_dict = {}
        for item in loaded_rated_items:
            if item is not None:
                score_assigned = float(items_scores_dict[item.content_id])
                if score_assigned >= threshold:
                    positive_rated_dict[item] = self.extract_features_item(item)

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(loaded_rated_items) == 0 or (loaded_rated_items.count(None) == len(loaded_rated_items)):
            raise NoRatedItems("User {} - No rated items available locally!".format(user_id))
        if len(positive_rated_dict) == 0:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!")

        self._positive_rated_dict = positive_rated_dict

    def fit(self):
        """
        The fit process for the CentroidVector consists in calculating the centroid of the features
        of the positive items ONLY.

        This method uses extracted features of the positive items stored in a private attribute, so
        `process_rated()` must be called before this method.

        The built centroid will also be stored in a private attribute.
        """
        positive_rated_features = list(self._positive_rated_dict.values())

        positive_rated_features_fused = self.fuse_representations(positive_rated_features, self._emb_combiner,
                                                                  as_array=True)
        self._centroid = positive_rated_features_fused.mean(axis=0)

        # we delete variable used to fit since will no longer be used
        del self._positive_rated_dict

    def predict(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                filter_list: List[str] = None) -> List[Interaction]:
        """
        CentroidVector is not a score prediction algorithm, calling this method will raise
        the `NotPredictionAlg` exception!

        Raises:
            NotPredictionAlg: exception raised since the CentroidVector algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("CentroidVector is not a Score Prediction Algorithm!")

    def rank(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
             recs_number: int = None, filter_list: List[str] = None) -> List[Interaction]:
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
        try:
            user_id = user_ratings[0].user_id
        except IndexError:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_seen_items = set([interaction.item_id for interaction in user_ratings])

        # Load items to predict
        if filter_list is None:
            items_to_predict = available_loaded_items.get_list([item_id for item_id in available_loaded_items
                                                                if item_id not in user_seen_items])
        else:
            items_to_predict = available_loaded_items.get_list(filter_list)

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Calculate predictions, they are the similarity of the new items with the centroid vector
            features_fused = self.fuse_representations(features_items_to_predict, self._emb_combiner, as_array=True)
            similarities = [self._similarity.perform(self._centroid, item) for item in features_fused]
        else:
            similarities = []

        # Build the item_score dict (key is item_id, value is rank score predicted)
        # and order the keys in descending order
        item_score_dict = dict(zip(id_items_to_predict, similarities))
        ordered_item_ids = sorted(item_score_dict, key=item_score_dict.get, reverse=True)

        # we only save the top-n items_ids corresponding to top-n recommendations
        # (if recs_number is None ordered_item_ids will contain all item_ids as the original list)
        ordered_item_ids = ordered_item_ids[:recs_number]

        # we construct the output data
        rank_interaction_list = [Interaction(user_id, item_id, item_score_dict[item_id])
                                 for item_id in ordered_item_ids]

        return rank_interaction_list

    def __str__(self):
        return "CentroidVector"

    def __repr__(self):
        return f'CentroidVector(item_field={self.item_field}, ' \
               f'similarity={self._similarity}, ' \
               f'threshold={self.threshold}, ' \
               f'embedding_combiner={self._emb_combiner})'
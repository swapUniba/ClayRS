from __future__ import annotations

from collections import defaultdict
from typing import List, Union, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer import Content
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.recsys.content_based_algorithm.classifier.classifiers import Classifier
    from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings

from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from clayrs.recsys.content_based_algorithm.content_based_algorithm import PerUserCBAlgorithm
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyPositiveItems, \
    OnlyNegativeItems, NotPredictionAlg, EmptyUserRatings


class ClassifierRecommender(PerUserCBAlgorithm):
    """
    Class that implements recommendation through a specified `Classifier` object.
    It's a ranking algorithm, so it can't do score prediction.

    Examples:
        * Interested in only a field representation, `DecisionTree` classifier from sklearn,
        `threshold` $= 3$ (Every item with rating score $>= 3$ will be considered as *positive*)

        >>> from clayrs import recsys as rs
        >>> alg = rs.ClassifierRecommender({"Plot": 0}, rs.SkDecisionTree(), 3)

        * Interested in only a field representation, `KNN` classifier with custom parameters from sklearn,
        `threshold` $= 3$ (Every item with rating score $>= 3$ will be considered as positive)

        >>> alg = rs.ClassifierRecommender({"Plot": 0}, rs.SkKNN(n_neighbors=3), 0)

        * Interested in multiple field representations of the items, `KNN` classifier with custom parameters from
        sklearn, `threshold` $= None$ (Every item with rating $>=$ mean rating of the user will be considered as positive)

        >>> alg = ClassifierRecommender(
        >>>                             item_field={"Plot": [0, "tfidf"],
        >>>                                         "Genre": [0, 1],
        >>>                                         "Director": "doc2vec"},
        >>>                             classifier=rs.SkKNN(n_neighbors=3),
        >>>                             threshold=None)

        !!! info

            After instantiating the ClassifierRecommender` algorithm, pass it in the initialization of
            a CBRS and the use its method to calculate ranking for single user or multiple users:

            Examples:

                >>> cbrs = rs.ContentBasedRS(algorithm=alg, ...)
                >>> cbrs.fit_rank(...)
                >>> # ...


    Args:
        item_field (dict): dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
        classifier (Classifier): classifier that will be used. Can be one object of the Classifier class.
        threshold: Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive. If the threshold is not specified, the average score of all items rated by the user is used.
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used, but they are in a
            matrix form instead of a single vector (e.g. WordEmbedding representations have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_classifier', '_embedding_combiner', '_labels', '_items_features')

    def __init__(self, item_field: dict, classifier: Classifier, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)

        self._classifier = classifier
        self._embedding_combiner = embedding_combiner
        self._labels: Optional[list] = None
        self._items_features: Optional[list] = None

    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            user_idx: Mapped integer of the active user (the user for which we must fit the algorithm)
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents

        Raises:
            EmptyUserRatings: Exception raised when the user does not appear in the train set
            NoRatedItems: Exception raised when there isn't any item available locally
                rated by the user
            OnlyPositiveItems: Exception raised when there are only positive items available locally
                for the user (Items that the user liked)
            OnlyNegativeitems: Exception raised when there are only negative items available locally
                for the user (Items that the user disliked)
        """

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

        threshold = self.threshold
        if threshold is None:
            threshold = np.nanmean(uir_user[:, 2])

        # Assign label and extract features from the rated items
        labels = []
        items_features = []

        # we extract feature of each item sorted based on its key: IMPORTANT for reproducibility!!
        # otherwise the matrix we feed to sklearn will have input item in different rows each run!
        for item in loaded_rated_items:
            if item is not None:

                score_assigned = map(float, items_scores_dict[item.content_id])

                for score in score_assigned:
                    items_features.append(self.extract_features_item(item))

                    if score >= threshold:
                        labels.append(1)
                    else:
                        labels.append(0)

        if len(uir_user[:, 1]) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        if len(items_features) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_idx))
        if 0 not in labels:
            raise OnlyPositiveItems("User {} - There are only positive items available locally!".format(user_idx))
        elif 1 not in labels:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!".format(user_idx))

        self._labels = labels
        self._items_features = items_features

    def fit_single_user(self):
        """
        Fit the classifier specified in the constructor with the features and labels
        extracted with the `process_rated()` method.

        It uses private attributes to fit the classifier, so `process_rated()` must be called
        before this method.
        """
        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(self._items_features, self._embedding_combiner)

        self._classifier.fit(fused_features, self._labels)

        # we delete variables used to fit since will no longer be used
        self._items_features = None
        self._labels = None

    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                            filter_list: List[str] = None) -> np.ndarray:
        """
        ClassifierRecommender is not a score prediction algorithm, calling this method will raise
        the `NotPredictionAlg` exception!

        Raises:
            NotPredictionAlg: exception raised since the ClassifierRecommender algorithm is not a
                score prediction algorithm
        """
        raise NotPredictionAlg("ClassifierRecommender is not a Score Prediction Algorithm!")

    def rank_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                         recs_number: Optional[int], filter_list: List[str]) -> np.ndarray:
        """
        Rank the top-n recommended items for the active user, where the top-n items to rank are controlled by the
        `recs_number` and `filter_list` parameter:

        * the former one is self-explanatory, the second is a list of items
        represented with their string ids. Must be necessarily strings and not their mapped integer since items are
        serialized following their string representation!

        If `recs_number` is `None`, all ranked items will be returned

        The filter list parameter is usually the result of the `filter_single()` method of a `Methodology` object

        Args:
            user_idx: Mapped integer of the active user
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            filter_list: list of the items to rank. Should contain string item ids

        Returns:
            uir matrix for a single user containing user and item idxs (integer representation) with the ranked score
                as third dimension sorted in a decreasing order
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

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features_items_to_pred = self.fuse_representations(features_items_to_predict,
                                                                 self._embedding_combiner)

        class_prob = self._classifier.predict_proba(fused_features_items_to_pred)

        # for each item we extract the probability that the item is liked (class 1)
        sorted_scores_idxs = np.argsort(class_prob[:, 1])[::-1][:recs_number]
        sorted_items = np.array(idx_items_to_predict)[sorted_scores_idxs]
        sorted_scores = class_prob[:, 1][sorted_scores_idxs]

        uir_rank = np.array([[user_idx, item_idx, score]
                             for item_idx, score in zip(sorted_items, sorted_scores)])

        return uir_rank

    def __str__(self):
        return "ClassifierRecommender"

    def __repr__(self):
        return f'ClassifierRecommender(item_field={self.item_field}, ' \
               f'classifier={self._classifier}, ' \
               f'threshold={self.threshold}, ' \
               f'embedding_combiner={self._embedding_combiner})'

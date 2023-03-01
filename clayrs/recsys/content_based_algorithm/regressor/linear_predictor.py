from __future__ import annotations
from collections import defaultdict
from typing import List, Union, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer import Content
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
    from clayrs.recsys.content_based_algorithm.regressor.regressors import Regressor
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings

from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    Centroid
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from clayrs.recsys.content_based_algorithm.content_based_algorithm import PerUserCBAlgorithm


class LinearPredictor(PerUserCBAlgorithm):
    """
    Class that implements recommendation through a specified linear predictor.
    It's a score prediction algorithm, so it can predict what rating a user would give to an unseen item.
    As such, it's also a ranking algorithm (it simply ranks in descending order unseen items by the predicted rating)

    Examples:

        * Interested in only a field representation, `LinearRegression` regressor from sklearn

        >>> from clayrs import recsys as rs
        >>> alg = rs.LinearPredictor({"Plot": 0}, rs.SkLinearRegression())

        * Interested in only a field representation, `Ridge` regressor from sklearn with custom parameters

        >>> alg = rs.LinearPredictor({"Plot": 0}, rs.SkRidge(alpha=0.8))

        * Interested in multiple field representations of the items, Ridge regressor from sklearn with custom
        parameters, `only_greater_eq` $= 2$ (Every item with rating $>= 2$ will be discarded and not considered in the
        ranking/score prediction task)

        >>> alg = rs.LinearPredictor(
        >>>                         item_field={"Plot": [0, "tfidf"],
        >>>                                     "Genre": [0, 1],
        >>>                                     "Director": "doc2vec"},
        >>>                         regressor=rs.SkRidge(alpha=0.8),
        >>>                         only_greater_eq=2)

        !!! info

            After instantiating the `LinearPredictor` algorithm, pass it in the initialization of
            a CBRS and the use its method to predict ratings or calculate ranking for a single user or multiple users:

            Examples:

                >>> cbrs = rs.ContentBasedRS(algorithm=alg, ...)
                >>> cbrs.fit_predict(...)
                >>> cbrs.fit_rank(...)
                >>> # ...

    Args:
        item_field: dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
        regressor: regressor that will be used. Can be one object of the `Regressor` class.
        only_greater_eq: Threshold for the ratings. Only items with rating greater or equal than the
            threshold will be considered, items with lower rating will be discarded. If None, no item will be filter out
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used, but they are in a
            matrix form instead of a single vector (e.g. WordEmbedding representations have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_regressor', '_labels', '_items_features', '_embedding_combiner')

    def __init__(self, item_field: dict, regressor: Regressor, only_greater_eq: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, only_greater_eq)
        self._regressor = regressor
        self._labels: Optional[list] = None
        self._items_features: Optional[list] = None
        self._embedding_combiner = embedding_combiner

    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the regressor.

        Features and labels (in this case the rating score) will be stored in private attributes of the class.

        IF there are no rated items available locally, an exception is thrown.

        Args:
            user_idx: Mapped integer of the active user (the user for which we must fit the algorithm)
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents

        Raises:
            EmptyUserRatings: Exception raised when the user does not appear in the train set
            NoRatedItems: Exception raised when there isn't any item available locally
                rated by the user
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

        # Assign label and extract features from the rated items
        labels = []
        items_features = []

        for item in loaded_rated_items:
            if item is not None:

                score_assigned = map(float, items_scores_dict[item.content_id])

                for score in score_assigned:
                    if self.threshold is None or score >= self.threshold:
                        items_features.append(self.extract_features_item(item))
                        labels.append(score)

        if len(uir_user[:, 1]) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        if len(items_features) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_idx))

        self._labels = labels
        self._items_features = items_features

    def fit_single_user(self):
        """
        Fit the regressor specified in the constructor with the features and labels (rating scores)
        extracted with the `process_rated()` method.

        It uses private attributes to fit the regressor, so `process_rated()` must be called
        before this method.
        """
        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(self._items_features, self._embedding_combiner)

        self._regressor.fit(fused_features, self._labels)

        # we delete variables used to fit since will no longer be used
        self._labels = None
        self._items_features = None

    def _common_prediction_process(self, user_idx: int, train_ratings: Ratings,
                                   available_loaded_items: LoadedContentsDict, filter_list: List[str] = None):
        """
        Simple private method which encapsulate common prediction process for both the `predict()` and `rank()`
        method, to avoid duplicate code
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
                # raises AttributeError if items are not present locally
                idx_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        idx_items_to_predict = train_ratings.item_map.convert_seq_str2int(idx_items_to_predict)

        if len(idx_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict,
                                                                     self._embedding_combiner)

            score_labels = self._regressor.predict(fused_features_items_to_pred)
        else:
            score_labels = []

        return idx_items_to_predict, score_labels

    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                            filter_list: List[str]) -> np.ndarray:
        """
        Predicts how much a user will like unrated items.

        The filter list parameter is usually the result of the `filter_single()` method of a `Methodology` object, and
        is a list of items represented with their string ids. Must be necessarily strings and not their mapped integer
        since items are serialized following their string representation!

        Args:
            user_idx: Mapped integer of the active user
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            filter_list: list of the items to rank. Should contain string item ids

        Returns:
            uir matrix for a single user containing user and item idxs (integer representation) with the predicted score
                as third dimension
        """

        idx_items_to_predict, score_labels = self._common_prediction_process(user_idx, train_ratings,
                                                                             available_loaded_items,
                                                                             filter_list)
        if len(score_labels) != 0:
            # Build the output data
            uir_pred = np.array(
                [[user_idx, item_idx, score] for item_idx, score in zip(idx_items_to_predict, score_labels)])
        else:
            uir_pred = np.array([])

        return uir_pred

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

        # Predict the rating for the items and sort them in descending order
        idx_items_to_predict, score_labels = self._common_prediction_process(user_idx, train_ratings,
                                                                             available_loaded_items,
                                                                             filter_list)

        if len(score_labels) != 0:
            sorted_scores_idxs = np.argsort(score_labels)[::-1][:recs_number]
            sorted_items = np.array(idx_items_to_predict)[sorted_scores_idxs]
            sorted_scores = score_labels[sorted_scores_idxs]

            # we construct the output data
            uir_rank = np.array([[user_idx, item_idx, score] for item_idx, score in zip(sorted_items, sorted_scores)])
        else:
            uir_rank = np.array([])

        return uir_rank

    def __str__(self):
        return "LinearPredictor"

    def __repr__(self):
        return f'LinearPredictor(item_field={self.item_field}, regressor={self._regressor}, ' \
               f'only_greater_eq={self.threshold}, embedding_combiner={self._embedding_combiner})'

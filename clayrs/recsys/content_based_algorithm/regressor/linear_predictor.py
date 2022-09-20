from collections import defaultdict
from typing import List, Union, Optional

from clayrs.content_analyzer import Content
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from clayrs.recsys.content_based_algorithm.regressor.regressors import Regressor

from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm


class LinearPredictor(ContentBasedAlgorithm):
    """
    Class that implements recommendation through a specified linear predictor.
    It's a score prediction algorithm, so it can predict what rating a user would give to an unseen item.
    As such, it's also a ranking algorithm (it simply ranks in descending order unseen items by the predicted rating)

    Examples:

        * Interested in only a field representation, LinearRegression regressor from sklearn

        >>> from clayrs import recsys as rs
        >>> alg = rs.LinearPredictor({"Plot": 0}, rs.SkLinearRegression())

        * Interested in only a field representation, Ridge regressor from sklearn with custom parameters

        >>> alg = rs.LinearPredictor({"Plot": 0}, rs.SkRidge(alpha=0.8))

        * Interested in multiple field representations of the items, Ridge regressor from sklearn with custom
        parameters, $only_greater_eq = 2$ (Every item with rating $>= 2$ will be discarded and not considered in the
        ranking/score prediction task)

        >>> alg = rs.LinearPredictor(
        >>>                         item_field={"Plot": [0, "tfidf"],
        >>>                                     "Genre": [0, 1],
        >>>                                     "Director": "doc2vec"},
        >>>                         regressor=rs.SkRidge(alpha=0.8),
        >>>                         only_greater_eq=2)

        !!! info

            After instantiating the LinearPredictor algorithm, pass it in the initialization of
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
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used but they are in a
            matrix form instead of a single vector (e.g. when WordEmbedding representations must be used you have one
            vector for each word). By default the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_regressor', '_labels', '_items_features', '_embedding_combiner')

    def __init__(self, item_field: dict, regressor: Regressor, only_greater_eq: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, only_greater_eq)
        self._regressor = regressor
        self._labels: Optional[list] = None
        self._items_features: Optional[list] = None
        self._embedding_combiner = embedding_combiner

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels (in this case the rating score) will be stored in private attributes of the class.

        IF there are no rated items available locally, an exception is thrown.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents

        Raises:
            NoRatedItems: Exception raised when there isn't any item available locally
                rated by the user
        """
        # a list since there could be duplicate interaction (eg bootstrap partitioning)
        items_scores_dict = defaultdict(list)
        for interaction in user_ratings:
            items_scores_dict[interaction.item_id].append(interaction.score)

        items_scores_dict = dict(sorted(items_scores_dict.items()))  # sort dictionary based on key for reproducibility

        # Create list of all the available items that are useful for the user
        loaded_rated_items: List[Union[Content, None]] = available_loaded_items.get_list([item_id
                                                                                          for item_id
                                                                                          in items_scores_dict.keys()])

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

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(items_features) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_id))

        self._labels = labels
        self._items_features = items_features

    def fit(self):
        """
        Fit the regressor specified in the constructor with the features and labels (rating scores)
        extracted with the process_rated() method.

        It uses private attributes to fit the classifier, so process_rated() must be called
        before this method.
        """
        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(self._items_features, self._embedding_combiner)

        self._regressor.fit(fused_features, self._labels)

        # we delete variables used to fit since will no longer be used
        del self._labels
        del self._items_features

    def _common_prediction_process(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                                   filter_list: List[str] = None):

        user_seen_items = set([interaction.item_id for interaction in user_ratings])

        # Load items to predict
        if filter_list is None:
            items_to_predict = available_loaded_items.get_list([item_id
                                                                for item_id in available_loaded_items
                                                                if item_id not in user_seen_items])
        else:
            items_to_predict = available_loaded_items.get_list(filter_list)

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                # raises AttributeError if items are not present locally
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict,
                                                                     self._embedding_combiner)

            score_labels = self._regressor.predict(fused_features_items_to_pred)
        else:
            score_labels = []

        return id_items_to_predict, score_labels

    def predict(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                filter_list: List[str] = None) -> List[Interaction]:
        """
        Predicts how much a user will like unrated items.

        One can specify which items must be predicted with the filter_list parameter,
        in this case ONLY items in the filter_list will be predicted.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be predicted.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            filter_list: List of the items to predict, if None all unrated items for the user will be predicted

        Returns:
            List of Interactions object where the 'score' attribute is the rating predicted by the algorithm
        """
        try:
            user_id = user_ratings[0].user_id
        except IndexError:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        id_items_to_predict, score_labels = self._common_prediction_process(user_ratings, available_loaded_items,
                                                                            filter_list)

        # Build the output data
        pred_interaction_list = [Interaction(user_id, item_id, score)
                                 for item_id, score in zip(id_items_to_predict, score_labels)]

        return pred_interaction_list

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

        # Predict the rating for the items and sort them in descending order
        id_items_to_predict, score_labels = self._common_prediction_process(user_ratings, available_loaded_items,
                                                                            filter_list)

        # Build the item_score dict (key is item_id, value is rank score predicted)
        # and order the keys in descending order
        item_score_dict = dict(zip(id_items_to_predict, score_labels))
        ordered_item_ids = sorted(item_score_dict, key=item_score_dict.get, reverse=True)

        # we only save the top-n items_ids corresponding to top-n recommendations
        # (if recs_number is None ordered_item_ids will contain all item_ids as the original list)
        ordered_item_ids = ordered_item_ids[:recs_number]

        # we construct the output data
        rank_interaction_list = [Interaction(user_id, item_id, item_score_dict[item_id])
                                 for item_id in ordered_item_ids]

        return rank_interaction_list

    def __repr__(self):
        return f'LinearPredictor(item_field={self.item_field}, regressor={self._regressor}, ' \
               f'only_greater_eq={self.threshold}, embedding_combiner={self._embedding_combiner})'

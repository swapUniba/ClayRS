from collections import defaultdict
from typing import List, Union, Optional

from clayrs.content_analyzer import Content
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.classifier.classifiers import Classifier
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from clayrs.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyPositiveItems, \
    OnlyNegativeItems, NotPredictionAlg, EmptyUserRatings


class ClassifierRecommender(ContentBasedAlgorithm):
    """
    Class that implements recommendation through a specified `Classifier`.
    It's a ranking algorithm so it can't do score prediction.

    Examples:
        * Interested in only a field representation, DecisionTree classifier from sklearn,
        $threshold = 3$ (Every item with rating score $>= 3$ will be considered as *positive*)

        >>> from clayrs import recsys as rs
        >>> alg = rs.ClassifierRecommender({"Plot": 0}, rs.SkDecisionTree(), 3)

        * Interested in only a field representation, KNN classifier with custom parameters from sklearn,
        $threshold = 3$ (Every item with rating score $>= 3$ will be considered as positive)

        >>> alg = rs.ClassifierRecommender({"Plot": 0}, rs.SkKNN(n_neighbors=3), 0)

        * Interested in multiple field representations of the items, KNN classifier with custom parameters from
        sklearn, $threshold = None$ (Every item with rating $>=$ mean rating of the user will be considered as positive)

        >>> alg = ClassifierRecommender(
        >>>                             item_field={"Plot": [0, "tfidf"],
        >>>                                         "Genre": [0, 1],
        >>>                                         "Director": "doc2vec"},
        >>>                             classifier=rs.SkKNN(n_neighbors=3),
        >>>                             threshold=None)

        !!! info

            After instantiating the ClassifierRecommender algorithm, pass it in the initialization of
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
            as positive. If the threshold is not specified, the average score of all items liked by the user is used.
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used but they are in a
            matrix form instead of a single vector (e.g. when WordEmbedding representations must be used you have one
            vector for each word). By default the `Centroid` of the rows of the matrix is computed
    """
    __slots__ = ('_classifier', '_embedding_combiner', '_labels', '_items_features')

    def __init__(self, item_field: dict, classifier: Classifier, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)
        self._classifier = classifier
        self._embedding_combiner = embedding_combiner
        self._labels: Optional[list] = None
        self._items_features: Optional[list] = None

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated_items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents

        Raises:
            NoRatedItems: Exception raised when there isn't any item available locally
                rated by the user
            OnlyPositiveItems: Exception raised when there are only positive items available locally
                for the user (Items that the user liked)
            OnlyNegativeitems: Exception raised when there are only negative items available locally
                for the user (Items that the user disliked)
        """
        # a list since there could be duplicate interaction (eg bootstrap partitioning)
        items_scores_dict = defaultdict(list)
        for interaction in user_ratings:
            items_scores_dict[interaction.item_id].append(interaction.score)

        items_scores_dict = dict(sorted(items_scores_dict.items()))  # sort dictionary based on key for reproducibility

        # Load rated items from the path
        loaded_rated_items: List[Union[Content, None]] = available_loaded_items.get_list([item_id
                                                                                          for item_id
                                                                                          in items_scores_dict.keys()])

        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

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

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(items_features) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_id))
        if 0 not in labels:
            raise OnlyPositiveItems("User {} - There are only positive items available locally!".format(user_id))
        elif 1 not in labels:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!".format(user_id))

        self._labels = labels
        self._items_features = items_features

    def fit(self):
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

    def predict(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                filter_list: List[str] = None) -> List[Interaction]:
        """
        ClassifierRecommender is not a score prediction algorithm, calling this method will raise
        the `NotPredictionAlg` exception!

        Raises:
            NotPredictionAlg: exception raised since the CentroidVector algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("ClassifierRecommender is not a Score Prediction Algorithm!")

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
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict, self._embedding_combiner)

            class_prob = self._classifier.predict_proba(fused_features_items_to_pred)
        else:
            class_prob = []

        # for each item we extract the probability that the item is liked (class 1)
        score_labels = (prob[1] for prob in class_prob)

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
        return f'ClassifierRecommender(item_field={self.item_field}, ' \
               f'classifier={self._classifier}, ' \
               f'threshold={self.threshold}, ' \
               f'embedding_combiner={self._embedding_combiner})'

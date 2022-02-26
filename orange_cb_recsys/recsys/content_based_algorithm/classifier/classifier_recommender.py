from typing import List, Union

import pandas as pd

from orange_cb_recsys.content_analyzer import Content
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Interaction, Prediction, Rank
from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import Classifier
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsDict
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyPositiveItems, \
    OnlyNegativeItems, NotPredictionAlg, EmptyUserRatings


class ClassifierRecommender(ContentBasedAlgorithm):
    """
    Class that implements recommendation through a specified Classifier.
    It's a ranking algorithm so it can't do score prediction.

    USAGE:
        > # Interested in only a field representation, DecisionTree classifier from sklearn,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = ClassifierRecommender({"Plot": 0}, SkDecisionTree(), 3)

        > # Interested in only a field representation, KNN classifier with custom parameters from sklearn,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = ClassifierRecommender({"Plot": 0}, SkKNN(n_neighbors=3), 0)

        > # Interested in multiple field representations of the items, KNN classifier with custom parameters from
        > # sklearn, threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = ClassifierRecommender(
        >                             item_field={"Plot": [0, "tfidf"],
        >                                         "Genre": [0, 1],
        >                                         "Director": "doc2vec"},
        >                             classifier=SkKNN(n_neighbors=3),
        >                             threshold=3)

        > # After instantiating the ClassifierRecommender algorithm, pass it in the initialization of
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
            classifier (Classifier): classifier that will be used.
               Can be one object of the Classifier class.
            threshold (float): Threshold for the ratings. If the rating is greater than the threshold, it will be considered
                as positive
       """

    def __init__(self, item_field: dict, classifier: Classifier, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)
        self.__classifier = classifier
        self.__embedding_combiner = embedding_combiner
        self.__labels: list = None
        self.__rated_dict: dict = None

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated_items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        Raises:
            NoRatedItems: exception raised when there isn't any item available locally
                rated by the user
            OnlyPositiveItems: exception raised when there are only positive items available locally
                for the user (Items that the user liked)
            OnlyNegativeitems: exception raised when there are only negative items available locally
                for the user (Items that the user disliked)
        """
        # Load rated items from the path
        items_scores_dict = {interaction.item_id: interaction.score for interaction in user_ratings}

        # Load rated items from the path
        loaded_rated_items: List[Union[Content, None]] = [available_loaded_items.get(item_id)
                                                          for item_id in set(items_scores_dict.keys())]

        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Assign label and extract features from the rated items
        labels = []
        rated_dict = {}

        for item in loaded_rated_items:
            if item is not None:
                rated_dict[item] = self.extract_features_item(item)

                # This conversion raises Exception when there are multiple same to_id for the user
                score_assigned = float(items_scores_dict[item.content_id])
                if score_assigned >= threshold:
                    labels.append(1)
                else:
                    labels.append(0)

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(rated_dict) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_id))
        if 0 not in labels:
            raise OnlyPositiveItems("User {} - There are only positive items available locally!".format(user_id))
        elif 1 not in labels:
            raise OnlyNegativeItems("User {} - There are only negative items available locally!".format(user_id))

        self.__labels = labels
        self.__rated_dict = rated_dict

    def fit(self):
        """
        Fit the classifier specified in the constructor with the features and labels
        extracted with the process_rated() method.

        It uses private attributes to fit the classifier, so process_rated() must be called
        before this method.
        """
        self._set_transformer()

        rated_features = list(self.__rated_dict.values())

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(rated_features, self.__embedding_combiner)

        self.__classifier.fit(fused_features, self.__labels)

    def predict(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                filter_list: List[str] = None) -> List[Interaction]:
        """
        ClassifierRecommender is not a score prediction algorithm, calling this method will raise
        the 'NotPredictionAlg' exception!
        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("ClassifierRecommender is not a Score Prediction Algorithm!")

    def rank(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
             recs_number: int = None, filter_list: List[str] = None) -> List[Interaction]:
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
        try:
            user_id = user_ratings[0].user_id
        except IndexError:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_seen_items = set([interaction.item_id for interaction in user_ratings])

        # Load items to predict
        if filter_list is None:
            items_to_predict = [available_loaded_items.get(item_id)
                                for item_id in available_loaded_items if item_id not in user_seen_items]
        else:
            items_to_predict = [available_loaded_items.get(item_id) for item_id in filter_list]

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            if item is not None:
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict, self.__embedding_combiner)

            class_prob = self.__classifier.predict_proba(fused_features_items_to_pred)
        else:
            class_prob = []

        # for each item we extract the probability that the item is liked (class 1)
        score_labels = [prob[1] for prob in class_prob]

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

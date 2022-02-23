from typing import List, Union

from orange_cb_recsys.content_analyzer import Content
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import Interaction, Ratings
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsIndex, LoadedContentsDict
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import Regressor
import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm


class LinearPredictor(ContentBasedAlgorithm):
    """
    Class that implements recommendation through a specified linear predictor.
    It's a score prediction algorithm, so it can predict what rating a user would give to an unseen item.
    As such, it's also a ranking algorithm (it simply ranks in descending order unseen items by the predicted rating)

    USAGE:
        > # Interested in only a field representation, LinearRegression regressor from sklearn,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = LinearPredictor({"Plot": 0}, SkLinearRegression(), 3)

        > # Interested in only a field representation, Ridge regressor from sklearn with custom parameters,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = LinearPredictor({"Plot": 0}, SkRidge(alpha=0.8), 0)

        > # Interested in multiple field representations of the items, Ridge regressor from sklearn with custom
        > # parameters, threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = LinearPredictor(
        >                       item_field={"Plot": [0, "tfidf"],
        >                                   "Genre": [0, 1],
        >                                   "Director": "doc2vec"},
        >                       regressor=SkRidge(alpha=0.8),
        >                       threshold=3)

        > # After instantiating the LinearPredictor algorithm, pass it in the initialization of
        > # a CBRS and the use its method to predict ratings or calculate ranking for a single user or multiple users:
        > cbrs = ContentBasedRS(algorithm=alg, ...)
        > cbrs.fit_predict(...)
        > cbrs.fit_rank(...)
        > ...
        > # Check the corresponding method documentation for more

       Args:
            item_field (dict): dict where the key is the name of the field
                that contains the content to use, value is the representation(s) id(s) that will be
                used for the said item. The value of a field can be a string or a list,
                use a list if you want to use multiple representations for a particular field.
            regressor (Regressor): classifier that will be used.
               Can be one object of the Classifier class.
            only_greater_eq (float): Threshold for the ratings. Only ratings which are greater or equal than the
                threshold will be considered, ratings which are less than the threshold will be discarded
       """

    def __init__(self, item_field: dict, regressor: Regressor, only_greater_eq: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, only_greater_eq)
        self.__regressor = regressor
        self.__labels: list = None
        self.__rated_dict: dict = None
        self.__embedding_combiner = embedding_combiner

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated_items available locally, an exception is thrown.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        items_scores_dict = {interaction.item_id: interaction.score for interaction in user_ratings}

        # Create list of all the available items that are useful for the user
        loaded_rated_items: List[Union[Content, None]] = [available_loaded_items.get(item_id)
                                                          for item_id in set(items_scores_dict.keys())]

        # Assign label and extract features from the rated items
        labels = []
        rated_dict = {}

        for item in loaded_rated_items:
            if item is not None:
                # This conversion raises Exception when there are multiple equals 'to_id' for the user
                score_assigned = float(items_scores_dict[item.content_id])

                if self.threshold is None:
                    rated_dict[item] = self.extract_features_item(item)
                    labels.append(score_assigned)
                elif score_assigned >= self.threshold:
                    rated_dict[item] = self.extract_features_item(item)
                    labels.append(score_assigned)

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(rated_dict) == 0:
            raise NoRatedItems("User {} - No rated item available locally!".format(user_id))

        self.__labels = labels
        self.__rated_dict = rated_dict

    def fit(self):
        """
        Fit the regressor specified in the constructor with the features and labels
        extracted with the process_rated() method.

        It uses private attributes to fit the classifier, so process_rated() must be called
        before this method.
        """
        self._set_transformer()

        rated_features = list(self.__rated_dict.values())

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(rated_features, self.__embedding_combiner)

        self.__regressor.fit(fused_features, self.__labels)


    def _common_prediction_process(self,user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
                                   filter_list: List[str] = None):

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
                # raises AttributeError if items are not present locally
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        if len(id_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict,
                                                                     self.__embedding_combiner)

            score_labels = self.__regressor.predict(fused_features_items_to_pred)
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
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
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

        predictions = Ratings.from_list(pred_interaction_list)

        return predictions

    def rank(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsDict,
             recs_number: int = None, filter_list: List[str] = None) -> List[Interaction]:
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

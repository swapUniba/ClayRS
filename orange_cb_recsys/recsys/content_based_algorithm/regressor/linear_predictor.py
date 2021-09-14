import logging
from typing import List

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, EmptyUserRatings
from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import Regressor
from orange_cb_recsys.utils.load_content import get_rated_items, get_unrated_items, \
    get_chosen_items
import pandas as pd


from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.utils.const import recsys_logger


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

    def process_rated(self, user_ratings: pd.DataFrame, items_directory: str):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated_items available locally, an exception is thrown.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        # Load rated items from the path
        rated_items = get_rated_items(items_directory, user_ratings)

        # Assign label and extract features from the rated items
        labels = []
        rated_dict = {}

        recsys_logger.info("Processing rated items")
        for item in rated_items:
            if item is not None:
                # This conversion raises Exception when there are multiple equals 'to_id' for the user
                score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)

                if self.threshold is None:
                    rated_dict[item] = self.extract_features_item(item)
                    labels.append(score_assigned)
                elif score_assigned >= self.threshold:
                    rated_dict[item] = self.extract_features_item(item)
                    labels.append(score_assigned)

        if user_ratings.empty:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings.from_id.iloc[0]
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
        recsys_logger.info("Fitting {} regressor".format(self.__regressor))
        self._set_transformer()

        rated_features = list(self.__rated_dict.values())

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(rated_features, self.__embedding_combiner)

        self.__regressor.fit(fused_features, self.__labels)

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
                # raises AttributeError if items are not present locally
                id_items_to_predict.append(item.content_id)
                features_items_to_predict.append(self.extract_features_item(item))

        recsys_logger.info("Calculating score predictions")
        if len(id_items_to_predict) > 0:
            # Fuse the input if there are dicts, multiple representation, etc.
            fused_features_items_to_pred = self.fuse_representations(features_items_to_predict, self.__embedding_combiner)

            score_labels = self.__regressor.predict(fused_features_items_to_pred)
        else:
            score_labels = []

        # Build the score_frame to return
        columns = ["to_id", "score"]
        score_frame = pd.DataFrame(columns=columns)

        score_frame["to_id"] = id_items_to_predict
        score_frame["score"] = score_labels

        return score_frame

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
        recsys_logger.info("Calculating rank")
        # we get the precedent level of the logger, so we will re-enable it at that level
        precedent_level_recsys_logger = recsys_logger.getEffectiveLevel()
        recsys_logger.setLevel(logging.WARNING)

        # Predict the rating for the items and sort them in descending order
        result = self.predict(user_ratings, items_directory, filter_list)

        result.sort_values(by=['score'], ascending=False, inplace=True)

        rank = result.head(recs_number)

        recsys_logger.setLevel(precedent_level_recsys_logger)
        return rank

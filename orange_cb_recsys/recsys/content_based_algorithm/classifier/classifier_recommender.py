from typing import List

import pandas as pd

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique, Centroid
from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import Classifier
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyPositiveItems, \
    OnlyNegativeItems, NotPredictionAlg
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import get_rated_items, get_unrated_items, load_content_instance


class ClassifierRecommender(ContentBasedAlgorithm):
    """
       Class that implements recommendation through a specified Classifier.

       In the constructor must be specified parameter needed for the recommendations.
       After instantiating the Recommender, the initialize() method of the superclass MUST BE CALLED!!
       Check the initialize() method documentation to see what need to be passed.

       The usage of the recommender is automated using the RecSys class (including the initialized part),
       but one can use the algorithm manually
        EXAMPLE:
            # Interested in only a field representation, DecisionTree classifier,
            # threshold = 0
            alg = ClassifierRecommender({"Plot": "0"}, DecisionTree(), 0)
            alg.initialize(...)

            # Interested in only a field representation, KNN classifier with custom parameter,
            # threshold = 0
            alg = ClassifierRecommender({"Plot": "0"}, KNN(n_neighbors=3), 0)
            alg.initialize(...)

            # Interested in multiple field representations of the items, KNN classifier with custom parameter,
            # threshold = 0
            alg = ClassifierRecommender(
                                        item_field={"Plot": ["0", "1"],
                                                    "Genre": ["0", "1"],
                                                    "Director": "1"},
                                        classifier=KNN(n_neighbors=3),
                                        threshold=0 )
            alg.initialize(...)

            # After instantiating and initializing the ClassifierRecommender, call the superclass method
            # calc_prediction() or calc_ranking() to calculate recommendations.
            # Check the corresponding method documentation for more
            alg.calc_prediction('U1', filter_list=['i1', 'i2])
            alg.calc_ranking('U1', recs_number=5)

       Args:
           item_field (dict): dict where the key is the name of the field
                that contains the content to use, value is the representation(s) that will be
                used for the said item. The value of a field can be a string or a list,
                use a list if you want to use multiple representations for a particular field.
                Check the example above for more.
           classifier (Classifier): classifier that will be used.
               Can be one object of the Classifier class.
           threshold (float): ratings bigger than threshold will be
               considered as positive
       """

    def __init__(self, item_field: dict, classifier: Classifier, threshold: float = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, threshold)
        self.__classifier = classifier
        self.__embedding_combiner = embedding_combiner
        self.__labels: list = None
        self.__rated_dict: dict = None

    def process_rated(self, user_ratings: pd.DataFrame, items_directory: str):
        """
        Function that extracts features from rated item and labels them.
        The extracted features will be later used to fit the classifier.

        Features and labels will be stored in private attributes of the class.

        IF there are no rated_items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        # Load rated items from the path
        rated_items = get_rated_items(items_directory, user_ratings)

        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Assign label and extract features from the rated items
        labels = []
        rated_dict = {}

        for item in rated_items:
            if item is not None:
                rated_dict[item] = self.extract_features_item(item)

                # This conversion raises Exception when there are multiple same to_id for the user
                score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)
                if score_assigned >= threshold:
                    labels.append(1)
                else:
                    labels.append(0)

        if len(rated_dict) == 0:
            raise NoRatedItems("No rated item available locally!\n"
                               "The score frame will be empty for the user")
        if 0 not in labels:
            raise OnlyPositiveItems("There are only positive items available locally!\n"
                                    "The score frame will be empty for the user")
        elif 1 not in labels:
            raise OnlyNegativeItems("There are only negative items available locally!\n"
                                    "The score frame will be empty for the user")

        self.__labels = labels
        self.__rated_dict = rated_dict

    def fit(self):
        """
        Fit the classifier specified in the constructor with the features and labels
        extracted with the process_rated() method.

        It uses private attributes to fit the classifier, that's why the method expects no parameter.
        """
        self._set_transformer()

        rated_features = list(self.__rated_dict.values())

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(rated_features, self.__embedding_combiner)

        self.__classifier.fit(fused_features, self.__labels)

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
        raise NotPredictionAlg("ClassifierRecommender is not a Score Prediction Algorithm!")

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
        # Load items to predict
        if filter_list is None:
            items_to_predict = get_unrated_items(items_directory, user_ratings)
        else:
            items_to_predict = [load_content_instance(items_directory, item_id) for item_id in filter_list]

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

            logger.info("Predicting scores")
            score_labels = self.__classifier.predict_proba(fused_features_items_to_pred)
        else:
            score_labels = []

        result = {'to_id': [], 'score': []}

        for item_id, score in zip(id_items_to_predict, score_labels):
            result['to_id'].append(item_id)
            result['score'].append(score[1])

        result = pd.DataFrame(result, columns=['to_id', 'score'])

        result.sort_values(by=['score'], ascending=False, inplace=True)

        rank = result[:recs_number]

        return rank

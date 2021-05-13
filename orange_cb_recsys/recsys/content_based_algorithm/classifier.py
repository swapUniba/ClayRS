import abc
import collections
from abc import ABC
from typing import List

from sklearn import neighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NoRatedItems, OnlyPositiveItems, \
    OnlyNegativeItems
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import get_rated_items, get_unrated_items, load_content_instance


class Classifier(ABC):
    """
    Abstract class for Classifiers

    The only concrete method is transform(). It has an abstract fit() method and an abstract
    predict_proba() method.
    """

    def __init__(self, **classifier_parameters):
        self.__transformer = DictVectorizer(sparse=True, sort=False)
        self.__classifier_parameters = classifier_parameters

    @property
    def classifier_parameters(self):
        return self.__classifier_parameters

    def is_parameters_empty(self):
        return len(self.classifier_parameters) == 0

    @abc.abstractmethod
    def fit(self, X: list, Y: list = None):
        """
        Fit the classifier.
        First the classifier is instantiated, then we transform the Training Data,
        then the actual fitting is done.

        Training data (X) is in the form:
            X = [ [representation1, representation2], [representation1, representation2], ...]
        where every sublist contains the representation chosen of the chosen fields for a item.

        Target data (Y) is in the form:
            Y = [0, 1, ... ]

        Args:
            X (list): list containing Training data.
            Y (list): list containing Training targets.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, X_pred: list):
        """
        Predicts the probability for every item in X_pred.
        First we transform the data, then the actual prediction is done.
        It uses the method predict_proba() from sklearn of the instantiated classifier

        It's in the form:
            X_pred = [ [representation1, representation2], [representation1, representation2], ...]
        where every sublist contains the representation chosen of the chosen fields for a item.

        Args:
            X_pred (list): list containing data to predict.
        """
        raise NotImplementedError


class KNN(Classifier):
    """
    Class that implements the KNN Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the Classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self, X: list):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.

        Since KNN has n_neighbors = 5 as default, it can throw an exception if less sample in
        the training data are provided, so we change dynamically the n_neighbors parameter
        according to the number of samples

        Args:
            X (list): Training data
        """
        if self.is_parameters_empty():
            if len(X) < 5:
                self.__clf = neighbors.KNeighborsClassifier(n_neighbors=len(X))
            else:
                self.__clf = neighbors.KNeighborsClassifier()
        else:
            self.__clf = neighbors.KNeighborsClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


class RandomForest(Classifier):
    """
    Class that implements the Random Forest Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.is_parameters_empty():
            self.__clf = RandomForestClassifier(n_estimators=400, random_state=42)
        else:
            self.__clf = RandomForestClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


class SVM(Classifier):
    """
    Class that implements the SVM Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier SVC directly from sklearn.

    The particularity is that if folds can be executed and are calculated with the
    method calc_folds(), then a Calibrated SVC classifier is instantiated.
    Otherwise a simple SVC classifier is instantiated.
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None
        self.__folds = None

    def calc_folds(self, labels: list):
        """
        Private functions that check what number of folds should SVM classifier do.

        By default SVM does 5 folds, so if there are less ratings we decrease the number of
        folds because it would throw an exception otherwise.
        Every class should have min 2 rated items, otherwise no folds can be executed.

        EXAMPLE:
                labels = [1 1 0 1 0]

            We count how many different values there are in the list with
            collections.Counter(labels), so:
                count = {"1": 3, "0": 2} # There are 3 rated_items of class 1
                                        # and 2 rated_items of class 0

            Then we search the min value in the dict with min(count.values()):
                min_fold = 2

        Args:
            labels: list of labels of the rated_items
        Returns:
            Number of folds to do.

        """
        count = collections.Counter(labels)
        min_fold = min(count.values())

        if min_fold < 2:
            logger.warning("There's too few rating for a class! There needs to be at least 2!\n"
                           "No folds will be executed")
        elif min_fold >= 5:
            min_fold = 5

        self.__folds = min_fold

    def __instantiate_classifier(self, calibrated: bool = True):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.

        Args:
            calibrated (bool): If True, a calibrated svc classifier is instantiated.
                Otherwise, a non-calibrated svc classifier is instantiated
        """

        if calibrated:
            if self.is_parameters_empty():
                self.__clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True),
                    cv=self.__folds)

            else:
                self.__clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True, **self.classifier_parameters),
                    cv=self.__folds)
        else:
            if self.is_parameters_empty():
                self.__clf = SVC(kernel='linear', probability=True)
            else:
                self.__clf = SVC(kernel='linear', probability=True, **self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        # Try fitting the Calibrated classifier for better classification
        try:
            self.__instantiate_classifier(calibrated=True)

            pipe = make_pipeline(self.__clf)
            self.__pipe = pipe.fit(X, Y)

        # If exception instantiate a non-calibrated classifier, then fit
        except ValueError:
            self.__instantiate_classifier(calibrated=False)

            pipe = make_pipeline(self.__clf)
            self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


class LogReg(Classifier):
    """
    Class that implements the Logistic Regression Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.is_parameters_empty():
            self.__clf = LogisticRegression(random_state=42)
        else:
            self.__clf = LogisticRegression(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


class DecisionTree(Classifier):
    """
    Class that implements the Decision Tree Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.is_parameters_empty():
            self.__clf = DecisionTreeClassifier(random_state=42)
        else:
            self.__clf = DecisionTreeClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


class GaussianProcess(Classifier):
    """
    Class that implements the Gaussian Process Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.is_parameters_empty():
            self.__clf = GaussianProcessClassifier(random_state=42)
        else:
            self.__clf = GaussianProcessClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        return self.__pipe.predict_proba(X_pred)


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

    def __init__(self, item_field: dict, classifier: Classifier, threshold: float = 0):
        super().__init__(item_field, threshold)
        self.__classifier = classifier
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

        # Assign label and extract features from the rated items
        labels = []
        rated_dict = {}

        for item in rated_items:
            if item is not None:
                rated_dict[item] = self.extract_features_item(item)

                score_assigned = float(user_ratings[user_ratings['to_id'] == item.content_id].score)
                labels.append(self.assign_class(score_assigned))

        if len(labels) == 0:
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
        # If the classifier chosen is SVM we calc how many folds the classifier
        # can do. If no folds is possible, no folds will be executed
        if isinstance(self.__classifier, SVM):
            self.__classifier.calc_folds(self.__labels)

        rated_features = list(self.__rated_dict.values())

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features = self.fuse_representations(rated_features)

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
        # Load items to predict
        if filter_list is None:
            items_to_predict = get_unrated_items(items_directory, user_ratings)
        else:
            items_to_predict = [load_content_instance(items_directory, item_id) for item_id in filter_list]

        # Extract features of the items to predict
        id_items_to_predict = []
        features_items_to_predict = []
        for item in items_to_predict:
            id_items_to_predict.append(item.content_id)
            features_items_to_predict.append(self.extract_features_item(item))

        # Fuse the input if there are dicts, multiple representation, etc.
        fused_features_items_to_pred = self.fuse_representations(features_items_to_predict)

        logger.info("Predicting scores")
        score_labels = self.__classifier.predict_proba(fused_features_items_to_pred)

        # Build the score_frame to return
        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        for score, item_id in zip(score_labels, id_items_to_predict):
            if item_id is not None:
                score_frame = pd.concat(
                    [score_frame, pd.DataFrame.from_records([(item_id, score[1])], columns=columns)],
                    ignore_index=True)

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

        # Predict the rating for the items and sort them in descending order
        result = self.predict(user_ratings, items_directory, filter_list)

        result.sort_values(by=['rating'], ascending=False, inplace=True)

        rank = result.head(recs_number)

        return rank

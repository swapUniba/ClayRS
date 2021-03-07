import collections
from abc import ABC
from typing import List

import numpy as np
from scipy.sparse import hstack
from sklearn import neighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

import pandas as pd

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
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

    def transform(self, X: list):
        """
        Transform the X passed vectorizing if X contains dicts and merging
        multiple representations in a single one for every item in X.
        So if X = [
                    [dict, arr, arr]
                        ...
                    [dict, arr, arr]
                ]
        where every sublist contains multiple representation for a single item,
        the function returns:
        X = [
                arr,
                ...
                arr
            ]
        Where every row is the fused representation for the item

        Args:
            X (list): list that contains representations of the items

        Returns:
            X fused and vectorized

        """

        # We check if there are dicts as representation in the first element of X,
        # since the representations are the same for all elements in X we can check
        # for dicts only in one element
        need_vectorizer = any(isinstance(rep, dict) for rep in X[0])

        if need_vectorizer:
            # IF the transformer is not fitted then we are training the model
            try:
                check_is_fitted(self.__transformer)
            except NotFittedError:
                X_dicts = []
                for item in X:
                    for rep in item:
                        if isinstance(rep, dict):
                            X_dicts.append(rep)

                self.__transformer.fit(X_dicts)

            # In every case, we transform the input
            X_vectorized = []
            for sublist in X:
                single_list = []
                for item in sublist:
                    if isinstance(item, dict):
                        vector = self.__transformer.transform(item)
                        single_list.append(vector)
                    else:
                        single_list.append(item)
                X_vectorized.append(single_list)
        else:
            X_vectorized = X

        try:
            X_sparse = [hstack(sublist).toarray().flatten() for sublist in X_vectorized]
        except ValueError:
            X_sparse = [np.column_stack(sublist).flatten() for sublist in X_vectorized]

        return X_sparse

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
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
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
        if self.__empty_parameters:
            if len(X) < 5:
                self.__clf = neighbors.KNeighborsClassifier(n_neighbors=len(X))
            else:
                self.__clf = neighbors.KNeighborsClassifier()
        else:
            self.__clf = neighbors.KNeighborsClassifier(**self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier(X)

        # Transform the input if there are dicts, multiple representation, etc.
        X = super().transform(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        X_pred = super().transform(X_pred)

        return self.__pipe.predict_proba(X_pred)


class RandomForest(Classifier):
    """
    Class that implements the Random Forest Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """
    def __init__(self, **classifier_parameters):
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.__empty_parameters:
            self.__clf = RandomForestClassifier(n_estimators=400, random_state=42)
        else:
            self.__clf = RandomForestClassifier(**self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = super().transform(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = super().transform(X_pred)

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
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
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
            if self.__empty_parameters:
                self.__clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True),
                    cv=self.__folds)

            else:
                self.__clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True, **self.__classifier_parameters),
                    cv=self.__folds)
        else:
            if self.__empty_parameters:
                self.__clf = SVC(kernel='linear', probability=True)
            else:
                self.__clf = SVC(kernel='linear', probability=True, **self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        # Transform the input
        X = super().transform(X)

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
        X_pred = super().transform(X_pred)

        return self.__pipe.predict_proba(X_pred)


class LogReg(Classifier):
    """
    Class that implements the Logistic Regression Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """
    def __init__(self, **classifier_parameters):
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.__empty_parameters:
            self.__clf = LogisticRegression(random_state=42)
        else:
            self.__clf = LogisticRegression(**self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = super().transform(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = super().transform(X_pred)

        return self.__pipe.predict_proba(X_pred)


class DecisionTree(Classifier):
    """
    Class that implements the Decision Tree Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """
    def __init__(self, **classifier_parameters):
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.__empty_parameters:
            self.__clf = DecisionTreeClassifier(random_state=42)
        else:
            self.__clf = DecisionTreeClassifier(**self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = super().transform(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = super().transform(X_pred)

        return self.__pipe.predict_proba(X_pred)


class GaussianProcess(Classifier):
    """
    Class that implements the Gaussian Process Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """
    def __init__(self, **classifier_parameters):
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
        self.__clf = None
        self.__pipe = None

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.__empty_parameters:
            self.__clf = GaussianProcessClassifier(random_state=42)
        else:
            self.__clf = GaussianProcessClassifier(**self.__classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = super().transform(X)

        pipe = make_pipeline(self.__clf)
        self.__pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = super().transform(X_pred)

        return self.__pipe.predict_proba(X_pred)


class ClassifierRecommender(RankingAlgorithm):
    """
       Class that implements recommendation through a specified Classifier.
       In the constructor must be specified parameter needed for the recommendations.
       To effectively get the recommended items, the method predict() of this class
       must be called after instantiating the ClassifierRecommender.
        EXAMPLE:
            # Interested in only a field representation, DecisionTree classifier,
            # threshold = 0
             alg = ClassifierRecommender({"Plot": "0"}, DecisionTree(), 0)

            # Interested in only a field representation, KNN classifier with custom parameter,
            # threshold = 0
             alg = ClassifierRecommender({"Plot": "0"}, KNN(n_neighbors=3), 0)

            # Interested in multiple field representations of the items, KNN classifier with custom parameter,
            # threshold = 0
             alg = ClassifierRecommender(
                                    item_field={"Plot": ["0", "1"],
                                                "Genre": ["0", "1"],
                                                "Director": ["1"]},
                                    classifier=KNN(n_neighbors=3),
                                    threshold=0 )

            # After instantiating the ClassifierRecommender, call the method predict to get
            # recommendations. Check the predict() method documentation for more
             alg.predict('U1', rating, 1, path)

       Args:
           item_field (dict): dict where the key is the name of the field
                that contains the content to use, value is the representation(s) that will be
                used for the said item. The value of a field can be a string or a list,
                use a list if you want to use multiple representations for a particular field.
                EXAMPLE:
                    {'Plot': '0'}
                    {'Genre': '1'}
                    {'Plot': ['0','1'], 'Genre': '0', 'Director': ['0', '1', '2']}
           classifier (Classifier): classifier that will be used
               can be one object of the Classifier class.
           threshold (int): ratings bigger than threshold will be
               considered as positive
       """

    def __init__(self, item_field: dict, classifier: Classifier, threshold: int = -1):
        super().__init__('', '')
        self.item_field = item_field
        self.__classifier = classifier
        self.__threshold = threshold

    def __calc_unrated_baglist(self, unrated_items: list):
        """
        Private functions that extracts features from unrated_items available locally.

        If multiple representations of the item are specified in the constructor of the class,
        we extract feature from all of them, otherwise if a single representation is specified,
        we extract feature from that single representation.

        Args:
            unrated_items (list): unrated items available locally
        Returns:
            unrated_features_bag_list (list): list that contains features extracted
                                        from the unrated_items
        """

        unrated_features_bag_list = []

        for item in unrated_items:
            if item is not None:
                single_item_bag_list = []
                for item_field in self.item_field:
                    field_representations = self.item_field[item_field]

                    if isinstance(field_representations, str):
                        # We have only one representation
                        representation = field_representations
                        single_item_bag_list.append(
                            item.get_field(item_field).get_representation(representation).value
                        )
                    else:
                        for representation in field_representations:
                            single_item_bag_list.append(
                                item.get_field(item_field).get_representation(representation).value
                            )
            unrated_features_bag_list.append(single_item_bag_list)

        return unrated_features_bag_list

    def __calc_labels_rated_baglist(self, rated_items: list, ratings: pd.DataFrame, threshold: float):
        """
        Private functions that calculates labels of rated_items available locally and
        extracts features from them.

        For every rated_items available locally, if the rating given is >= threshold
        then we label it as 1, 0 otherwise.
        We also extract features from the rated items that we will use later to fit the
        classifier, from a single representation or from multiple ones.
        IF there are no rated_items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            rated_items (list): rated items by the user available locally
            ratings (DataFrame): Dataframe which contains ratings given by the user
            threshold (float): float that separates positive ratings from the negative ones
        Returns:
            labels (list): list of labels of the rated items
            rated_features_bag_list (list): list that contains features extracted
                                    from the rated_items
        """
        labels = []
        rated_features_bag_list = []

        for item in rated_items:
            if item is not None:
                single_item_bag_list = []
                for item_field in self.item_field:
                    field_representations = self.item_field[item_field]

                    if isinstance(field_representations, str):
                        # We have only one representation
                        representation = field_representations
                        single_item_bag_list.append(
                            item.get_field(item_field).get_representation(representation).value
                        )
                    else:
                        for representation in field_representations:
                            single_item_bag_list.append(
                                item.get_field(item_field).get_representation(representation).value
                            )

            labels.append(
                    1 if float(ratings[ratings['to_id'] == item.content_id].score) >= threshold else 0
            )
            rated_features_bag_list.append(single_item_bag_list)

        if len(labels) == 0:
            raise FileNotFoundError("No rated item available locally!\n"
                                    "The score frame will be empty for the user")
        if 0 not in labels:
            raise ValueError("There are only positive items available locally!\n"
                             "The score frame will be empty for the user")
        elif 1 not in labels:
            raise ValueError("There are only negative items available locally!\n"
                             "The score frame will be empty for the user")

        return labels, rated_features_bag_list

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None) -> pd.DataFrame:
        """
        Get recommendations for a specified user.

        You must pass the user_id, the DataFrame which contains the ratings of the user, how many
        recommended item the method predict() must return, and the path of the items.
        If recommendation for certain item is needed, specify them in candidate_item_id_list
        parameter. In this case, the recommender system will return only scores for the items
        in the list, ignoring the recs_number parameter.
         EXAMPLE
            # Instantiate the ClassifierRecommender object, check its documentation if needed
             alg = ClassifierRecommender(...)

            # Get 5 most recommended items for the user 'AOOO'
             alg.predict('A000', rat, 5, path)

            # Get the score for the item 'tt0114576' for the user 'A000'
             alg.predict('A000', ratings, 1, path, ['tt0114576'])

        Args:
            user_id (str): user for which recommendations will be computed
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            recs_number (int): How long the ranking will be
            items_directory (str): Path to the directory where the items are stored.
            candidate_item_id_list: list of the items that can be recommended, if None
            all unrated items will be used
        Returns:
            The predicted classes, or the predict values.
        """

        # Load unrated items from the path
        if candidate_item_id_list is None:
            unrated_items = get_unrated_items(items_directory, ratings)
        else:
            unrated_items = [load_content_instance(items_directory, item_id) for item_id in candidate_item_id_list]

        # Load rated items from the path
        rated_items = get_rated_items(items_directory, ratings)

        # If threshold is the min possible (range is [-1, 1]), we calculate the mean value
        # of all the ratings and set it as the threshold
        if self.__threshold == -1:
            threshold = pd.to_numeric(ratings["score"], downcast="float").mean()
        else:
            threshold = self.__threshold

        # Calculates labels and extract features from the rated items
        # If exception, returns an empty score_frame
        try:
            labels, rated_features_bag_list = \
                self.__calc_labels_rated_baglist(rated_items, ratings, threshold)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(str(e))
            columns = ["to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            return score_frame

        # Extract features from unrated items
        unrated_features_bag_list = self.__calc_unrated_baglist(unrated_items)

        # If the classifier chosen is SVM we calc how many folds the classifier
        # can do. If no folds is possible, no folds will be executed
        if isinstance(self.__classifier, SVM):
            self.__classifier.calc_folds(labels)

        self.__classifier.fit(rated_features_bag_list, labels)

        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        logger.info("Predicting scores")
        score_labels = self.__classifier.predict_proba(unrated_features_bag_list)

        for score, item in zip(score_labels, unrated_items):
            if item is not None:
                score_frame = pd.concat(
                    [score_frame, pd.DataFrame.from_records([(item.content_id, score[1])], columns=columns)],
                    ignore_index=True)

        score_frame = score_frame.sort_values(['rating'], ascending=False).reset_index(drop=True)
        score_frame = score_frame[:recs_number]

        return score_frame

from abc import ABC

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Classifier(ABC):
    """
    Abstract class for Classifiers
    """

    def __init__(self, classifier):
        self.__classifier = classifier

    @property
    def classifier(self):
        return self.__classifier

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
        self.classifier.fit(X, Y)

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
        return self.classifier.predict_proba(X_pred)


class SkSVC(Classifier):
    """
    Class that implements the SVC Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier SVC directly from sklearn, except for the 'probability' parameter:
    it is set at True and cannot be changed
    """
    def __init__(self, *args, **kwargs):
        # parameters_passed = locals().copy()
        # del parameters_passed['self']
        # del parameters_passed['__class__']

        # Force the probability parameter at True, otherwise SVC wont' predict_proba
        kwargs['probability'] = True
        clf = SVC(*args, **kwargs)

        super().__init__(clf)


class SkKNN(Classifier):
    """
    Instantiate the sklearn classifier.

    If parameters were passed in the constructor of this class, then
    thery are passed to sklearn.

    Since KNN has n_neighbors = 5 as default, it can throw an exception if less sample in
    the training data are provided, so we change dynamically the n_neighbors parameter
    according to the number of samples

    Args:
        X (list): Training data
    """
    def __init__(self, *args, **kwargs):

        clf = KNeighborsClassifier(*args, **kwargs)

        super().__init__(clf)

    def fit(self, X: list, Y: list = None):
        """
        Since KNN has n_neighbors = 5 as default, it can throw an exception if less sample in
        the training data are provided, so if the user didn't specify a custom 'n_neighbors' parameter,
        we change it dynamically according to the number of samples
        """
        # If the user did not pass a custom n_neighbor the algorithm tries to fix the error
        # if n_samples < n_neighbors
        if self.classifier.n_neighbors == 5 and len(X) < self.classifier.n_neighbors:
            self.classifier.n_neighbors = len(X)

        super().fit(X, Y)


class SkRandomForest(Classifier):
    """
    Class that implements the Random Forest Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """
    def __init__(self, *args, **kwargs):

        clf = RandomForestClassifier(*args, **kwargs)

        super().__init__(clf)


class SkLogisticRegression(Classifier):
    """
    Class that implements the Logistic Regression Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, *args, **kwargs):

        clf = LogisticRegression(*args, **kwargs)
        super().__init__(clf)


class SkDecisionTree(Classifier):
    """
    Class that implements the Decision Tree Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, *args, **kwargs):

        clf = DecisionTreeClassifier(*args, **kwargs)
        super().__init__(clf)


class SkGaussianProcess(Classifier):
    """
    Class that implements the Gaussian Process Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, *args, **kwargs):

        clf = GaussianProcessClassifier(*args, **kwargs)
        super().__init__(clf)
import inspect
from abc import ABC, abstractmethod
from typing import Union, Any

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from clayrs.utils.automatic_methods import autorepr


class Classifier(ABC):
    """
    Abstract class for Classifiers
    """

    def __init__(self, classifier, currentframe):
        self.__classifier = classifier

        self._repr_string = autorepr(self, currentframe)

    @property
    def classifier(self):
        return self.__classifier

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        """
        Fit the classifier.

        Training data (X) is in the form:

            `X = [ (merged) representation for item 1, (merged) representation for item 2, ...]`

        where every item is a representation for an item (can be a merged one in case multiple representations
        are chosen)

        Target data (Y) is in the form:
            Y = [0, 1, ... ]
        where 0 represent a negative item, 1 represent a positive item

        Args:
            X: list containing Training data.
            Y: list containing Training targets.
        """
        self.classifier.fit(X, Y)

    def predict_proba(self, X_pred: list):
        """
        Predicts the probability for every item in X_pred.
        It uses the method predict_proba() from sklearn of the instantiated classifier

        It's in the form:
            X_pred = [ (merged) representation for item 1, (merged) representation for item 2, ...]
        where every item is a representation for an item (can be a merged one in case multiple representations
        are chosen)

        Args:
            X_pred: list containing data to predict.
        """
        return self.classifier.predict_proba(X_pred)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return self._repr_string


class SkSVC(Classifier):
    """
    Class that implements the SVC Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier SVC directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

    The only parameter from sklearn that cannot be passed is the 'probability' parameter:
    it is set to True and cannot be changed
    """

    def __init__(self,
                 *,
                 C: Any = 1.0,
                 kernel: Any = "rbf",
                 degree: Any = 3,
                 gamma: Any = "scale",
                 coef0: Any = 0.0,
                 shrinking: Any = True,
                 tol: Any = 1e-3,
                 cache_size: Any = 200,
                 class_weight: Any = None,
                 verbose: Any = False,
                 max_iter: Any = -1,
                 decision_function_shape: Any = "ovr",
                 break_ties: Any = False,
                 random_state: Any = None):

        # Force the probability parameter at True, otherwise SVC won't predict_proba
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, tol=tol,
                  cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter,
                  decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state,
                  probability=True)

        super().__init__(clf, inspect.currentframe())

    def __str__(self):
        return "SkSVC"


class SkKNN(Classifier):
    """
    Class that implements the KNN Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier KNN directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

    Since KNN implementation of sklearn has `n_neighbors = 5` as default, it can throw an exception if less sample in
    the training data are provided, so we change dynamically the n_neighbors parameter
    according to the number of samples if the dataset is too small and if no manual `n_neighbors` is set
    """

    def __init__(self, n_neighbors: Any = 5,
                 *,
                 weights: Any = "uniform",
                 algorithm: Any = "auto",
                 leaf_size: Any = 30,
                 p: Any = 2,
                 metric: Any = "minkowski",
                 metric_params: Any = None,
                 n_jobs: Any = None):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
                                   p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)

        super().__init__(clf, inspect.currentframe())

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        # If the user did not pass a custom n_neighbor the algorithm tries to fix the error
        # if n_samples < n_neighbors
        if self.classifier.n_neighbors == 5 and X.shape[0] < self.classifier.n_neighbors:
            self.classifier.n_neighbors = X.shape[0]

        super().fit(X, Y)

    def __str__(self):
        return "SkKNN"


class SkRandomForest(Classifier):
    """
    Class that implements the Random Forest Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

    """

    def __init__(self, n_estimators: Any = 100,
                 *,
                 criterion: Any = "gini",
                 max_depth: Any = None,
                 min_samples_split: Any = 2,
                 min_samples_leaf: Any = 1,
                 min_weight_fraction_leaf: Any = 0.0,
                 max_features: Any = "auto",
                 max_leaf_nodes: Any = None,
                 min_impurity_decrease: Any = 0.0,
                 bootstrap: Any = True,
                 oob_score: Any = False,
                 n_jobs: Any = None,
                 random_state: Any = None,
                 verbose: Any = 0,
                 warm_start: Any = False,
                 class_weight: Any = None,
                 ccp_alpha: Any = 0.0,
                 max_samples: Any = None):
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                     max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                     bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                     verbose=verbose, warm_start=warm_start, class_weight=class_weight,
                                     ccp_alpha=ccp_alpha, max_samples=max_samples)

        super().__init__(clf, inspect.currentframe())

    def __str__(self):
        return "SkRandomForest"


class SkLogisticRegression(Classifier):
    """
    Class that implements the Logistic Regression Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
    """

    def __init__(self, penalty: Any = "l2",
                 *,
                 dual: Any = False,
                 tol: Any = 1e-4,
                 C: Any = 1.0,
                 fit_intercept: Any = True,
                 intercept_scaling: Any = 1,
                 class_weight: Any = None,
                 random_state: Any = None,
                 solver: Any = "lbfgs",
                 max_iter: Any = 100,
                 multi_class: Any = "auto",
                 verbose: Any = 0,
                 warm_start: Any = False,
                 n_jobs: Any = None,
                 l1_ratio: Any = None):
        clf = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                 intercept_scaling=intercept_scaling, class_weight=class_weight,
                                 random_state=random_state, solver=solver, max_iter=max_iter,
                                 multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                 n_jobs=n_jobs, l1_ratio=l1_ratio)

        super().__init__(clf, inspect.currentframe())

    def __str__(self):
        return "SkLogisticRegression"


class SkDecisionTree(Classifier):
    """
    Class that implements the Decision Tree Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    """

    def __init__(self, *,
                 criterion: Any = "gini",
                 splitter: Any = "best",
                 max_depth: Any = None,
                 min_samples_split: Any = 2,
                 min_samples_leaf: Any = 1,
                 min_weight_fraction_leaf: Any = 0.0,
                 max_features: Any = None,
                 random_state: Any = None,
                 max_leaf_nodes: Any = None,
                 min_impurity_decrease: Any = 0.0,
                 class_weight: Any = None,
                 ccp_alpha: Any = 0.0):
        clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                     random_state=random_state, max_leaf_nodes=max_leaf_nodes,
                                     min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
                                     ccp_alpha=ccp_alpha)

        super().__init__(clf, inspect.currentframe())

    def __str__(self):
        return "SkDecisionTree"


class SkGaussianProcess(Classifier):
    """
    Class that implements the Gaussian Process Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)
    """

    def __init__(self, kernel: Any = None,
                 *,
                 optimizer: Any = "fmin_l_bfgs_b",
                 n_restarts_optimizer: Any = 0,
                 max_iter_predict: Any = 100,
                 warm_start: Any = False,
                 copy_X_train: Any = True,
                 random_state: Any = None,
                 multi_class: Any = "one_vs_rest",
                 n_jobs: Any = None):

        clf = GaussianProcessClassifier(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                        max_iter_predict=max_iter_predict, warm_start=warm_start,
                                        copy_X_train=copy_X_train, random_state=random_state,
                                        multi_class=multi_class, n_jobs=n_jobs)

        super().__init__(clf, inspect.currentframe())

    def __str__(self):
        return "SkGaussianProcess"

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        self.classifier.fit(X.toarray() if isinstance(X, sparse.csr_matrix) else X, Y)

    def predict_proba(self, X_pred: Union[np.ndarray, sparse.csr_matrix]):
        return self.classifier.predict_proba(X_pred.toarray() if isinstance(X_pred, sparse.csr_matrix) else X_pred)

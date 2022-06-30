import inspect
from abc import ABC
from typing import Union, Any

import numpy as np
from scipy import sparse

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, SGDRegressor, ARDRegression, \
    HuberRegressor, PassiveAggressiveRegressor
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON

from clayrs.utils.automatic_methods import autorepr


class Regressor(ABC):
    """
    Abstract class for Regressors
    """

    def __init__(self, model, currentframe):
        self.__model = model

        self._repr_string = autorepr(self, currentframe)

    @property
    def model(self):
        return self.__model

    def fit(self, X: list, Y: list = None):
        """
        Fit the instantiated regressor.

        Training data (X) is in the form:

            `X = [ (merged) representation for item 1, (merged) representation for item 2, ...]`

        where every item is a representation for an item (can be a merged one in case multiple representations
        are chosen)

        Target data (Y) is in the form:
            `Y = [1.2, 4.0, ... ]`

        where all labels are the numeric score given by the user

        Args:
            X: list containing Training data.
            Y: list containing Training targets.
        """
        self.__model = self.__model.fit(X, Y)

    def predict(self, X_pred: list):
        """
        Predicts the score for every item in X_pred.
        It uses the method predict() from sklearn of the instantiated regressor

        It's in the form:
            `X_pred = [ (merged) representation for item 1, (merged) representation for item 2, ...]`

        where every item is a representation for an item (can be a merged one in case multiple representations
        are chosen)

        Args:
            X_pred: list containing data to predict.
        """
        return self.__model.predict(X_pred)

    def __repr__(self):
        return self._repr_string


class SkLinearRegression(Regressor):
    """
    Class that implements the LinearRegression regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor LinearRegression directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    """

    def __init__(self, *, fit_intercept: Any = True,
                 normalize: Any = "deprecated",
                 copy_X: Any = True,
                 n_jobs: Any = None,
                 positive: Any = False):
        model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs,
                                 positive=positive)

        super().__init__(model, inspect.currentframe())

    def __str__(self):
        return "SkLinearRegression"


class SkRidge(Regressor):
    """
    Class that implements the Ridge regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor Ridge directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

    """

    def __init__(self, alpha: Any = 1.0,
                 *,
                 fit_intercept: Any = True,
                 normalize: Any = "deprecated",
                 copy_X: Any = True,
                 max_iter: Any = None,
                 tol: Any = 1e-3,
                 solver: Any = "auto",
                 positive: Any = False,
                 random_state: Any = None):
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
                      max_iter=max_iter, tol=tol, solver=solver, positive=positive, random_state=random_state)

        super().__init__(model, inspect.currentframe())

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        self.model.fit(X.toarray() if isinstance(X, sparse.csr_matrix) else X, Y)

    def predict(self, X_pred: Union[np.ndarray, sparse.csr_matrix]):
        return self.model.predict(X_pred.toarray() if isinstance(X_pred, sparse.csr_matrix) else X_pred)

    def __str__(self):
        return "SkRidge"


class SkBayesianRidge(Regressor):
    """
    Class that implements the BayesianRidge regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor BayesianRidge directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)

    """

    def __init__(self, *,
                 n_iter: Any = 300,
                 tol: Any = 1.0e-3,
                 alpha_1: Any = 1.0e-6,
                 alpha_2: Any = 1.0e-6,
                 lambda_1: Any = 1.0e-6,
                 lambda_2: Any = 1.0e-6,
                 alpha_init: Any = None,
                 lambda_init: Any = None,
                 compute_score: Any = False,
                 fit_intercept: Any = True,
                 normalize: Any = "deprecated",
                 copy_X: Any = True,
                 verbose: Any = False):
        model = BayesianRidge(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                              lambda_2=lambda_2, alpha_init=alpha_init, lambda_init=lambda_init,
                              compute_score=compute_score, fit_intercept=fit_intercept, normalize=normalize,
                              copy_X=copy_X, verbose=verbose)

        super().__init__(model, inspect.currentframe())

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        self.model.fit(X.toarray() if isinstance(X, sparse.csr_matrix) else X, Y)

    def predict(self, X_pred: Union[np.ndarray, sparse.csr_matrix]):
        return self.model.predict(X_pred.toarray() if isinstance(X_pred, sparse.csr_matrix) else X_pred)

    def __str__(self):
        return "SkBayesianRidge"


class SkSGDRegressor(Regressor):
    """
    Class that implements the SGD regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor SGD directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
    """

    def __init__(self, loss: Any = "squared_error",
                 *,
                 penalty: Any = "l2",
                 alpha: Any = 0.0001,
                 l1_ratio: Any = 0.15,
                 fit_intercept: Any = True,
                 max_iter: Any = 1000,
                 tol: Any = 1e-3,
                 shuffle: Any = True,
                 verbose: Any = 0,
                 epsilon: Any = DEFAULT_EPSILON,
                 random_state: Any = None,
                 learning_rate: Any = "invscaling",
                 eta0: Any = 0.01,
                 power_t: Any = 0.25,
                 early_stopping: Any = False,
                 validation_fraction: Any = 0.1,
                 n_iter_no_change: Any = 5,
                 warm_start: Any = False,
                 average: Any = False):
        model = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
                             max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon,
                             random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t,
                             early_stopping=early_stopping, validation_fraction=validation_fraction,
                             n_iter_no_change=n_iter_no_change, warm_start=warm_start, average=average)
        super().__init__(model, inspect.currentframe())

    def __str__(self):
        return "SkSGDRegressor"


class SkARDRegression(Regressor):
    """
    Class that implements the ARD regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor ARD directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)
    """

    def __init__(self, *,
                 n_iter: Any = 300,
                 tol: Any = 1.0e-3,
                 alpha_1: Any = 1.0e-6,
                 alpha_2: Any = 1.0e-6,
                 lambda_1: Any = 1.0e-6,
                 lambda_2: Any = 1.0e-6,
                 compute_score: Any = False,
                 threshold_lambda: Any = 1.0e4,
                 fit_intercept: Any = True,
                 normalize: Any = "deprecated",
                 copy_X: Any = True,
                 verbose: Any = False):
        model = ARDRegression(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                              lambda_2=lambda_2, compute_score=compute_score, threshold_lambda=threshold_lambda,
                              fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, verbose=verbose)
        super().__init__(model, inspect.currentframe())

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], Y: list = None):
        self.model.fit(X.toarray() if isinstance(X, sparse.csr_matrix) else X, Y)

    def predict(self, X_pred: Union[np.ndarray, sparse.csr_matrix]):
        return self.model.predict(X_pred.toarray() if isinstance(X_pred, sparse.csr_matrix) else X_pred)

    def __str__(self):
        return "SkARDRegression"


class SkHuberRegressor(Regressor):
    """
    Class that implements the Huber regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor Huber directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
    """

    def __init__(self, *,
                 epsilon: Any = 1.35,
                 max_iter: Any = 100,
                 alpha: Any = 0.0001,
                 warm_start: Any = False,
                 fit_intercept: Any = True,
                 tol: Any = 1e-05):
        model = HuberRegressor(epsilon=epsilon, max_iter=max_iter, alpha=alpha,
                               warm_start=warm_start, fit_intercept=fit_intercept, tol=tol)

        super().__init__(model, inspect.currentframe())

    def __str__(self):
        return "SkHuberRegressor"


class SkPassiveAggressiveRegressor(Regressor):
    """
    Class that implements the PassiveAggressive regressor from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the regressor PassiveAggressive directly from sklearn.

    Sklearn documentation: [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)
    """

    def __init__(self, *,
                 C: Any = 1.0,
                 fit_intercept: Any = True,
                 max_iter: Any = 1000,
                 tol: Any = 1e-3,
                 early_stopping: Any = False,
                 validation_fraction: Any = 0.1,
                 n_iter_no_change: Any = 5,
                 shuffle: Any = True,
                 verbose: Any = 0,
                 loss: Any = "epsilon_insensitive",
                 epsilon: Any = DEFAULT_EPSILON,
                 random_state: Any = None,
                 warm_start: Any = False,
                 average: Any = False):

        model = PassiveAggressiveRegressor(C=C, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
                                           early_stopping=early_stopping, validation_fraction=validation_fraction,
                                           n_iter_no_change=n_iter_no_change, shuffle=shuffle, verbose=verbose,
                                           loss=loss, epsilon=epsilon, random_state=random_state, warm_start=warm_start,
                                           average=average)
        super().__init__(model, inspect.currentframe())

    def __str__(self):
        return "SkPassiveAggressiveRegressor"

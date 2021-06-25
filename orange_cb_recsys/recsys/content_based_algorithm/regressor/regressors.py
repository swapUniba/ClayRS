from abc import ABC
from typing import Union

from sklearn.linear_model._base import LinearModel as SKLinearModel
from sklearn.linear_model._stochastic_gradient import BaseSGDRegressor

from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, SGDRegressor, ARDRegression,\
    HuberRegressor, PassiveAggressiveRegressor


class Regressor(ABC):
    """
    Abstract class for Classifiers

    The only concrete method is transform(). It has an abstract fit() method and an abstract
    predict_proba() method.
    """

    def __init__(self, model: Union[SKLinearModel, BaseSGDRegressor]):
        self.__model = model

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
        self.__model = self.__model.fit(X, Y)

    def predict(self, X_pred: list):
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
        return self.__model.predict(X_pred)


class SkLinearRegression(Regressor):

    def __init__(self, *args, **kwargs):
        model = LinearRegression(*args, **kwargs)

        super().__init__(model)


class SkRidge(Regressor):
    def __init__(self, *args, **kwargs):
        model = Ridge(*args, **kwargs)

        super().__init__(model)


class SkBayesianRidge(Regressor):
    def __init__(self, *args, **kwargs):
        model = BayesianRidge(*args, **kwargs)

        super().__init__(model)


class SkSGDRegressor(Regressor):
    def __init__(self, *args, **kwargs):
        model = SGDRegressor(*args, **kwargs)
        super().__init__(model)


class SkARDRegression(Regressor):
    def __init__(self, *args, **kwargs):

        model = ARDRegression(*args, **kwargs)
        super().__init__(model)


class SkHuberRegressor(Regressor):
    def __init__(self, *args, **kwargs):

        model = HuberRegressor(*args, **kwargs)
        super().__init__(model)


class SkPassiveAggressiveRegressor(Regressor):
    def __init__(self, *args, **kwargs):

        model = PassiveAggressiveRegressor(*args, **kwargs)
        super().__init__(model)

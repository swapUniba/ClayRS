import abc
from abc import ABC


class Algorithm(ABC):
    """
    Abstract class for an Algorithm.

    Every algorithm must be able to predict or to rank, or maybe both.
    In case some algorithms can only do one of the two (eg. PageRank), simply implement both
    methods and raise the NotPredictionAlg or NotRankingAlg exception accordingly.
    """
    __slots__ = ()

    @abc.abstractmethod
    def predict(self, **kwargs):
        """
        Method to call when score prediction must be done.

        If the Algorithm can't do score prediction, implement this method and raise
        the NotPredictionAlg exception
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, **kwargs):
        """
        Method to call when ranking must be done.

        If the Algorithm can't rank, implement this method and raise the NotRankingAlg exception
        """
        raise NotImplementedError

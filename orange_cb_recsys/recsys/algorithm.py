import abc
from abc import ABC


class Algorithm(ABC):
    """
    Abstract class for an Algorithm
    """

    @abc.abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, **kwargs):
        raise NotImplementedError

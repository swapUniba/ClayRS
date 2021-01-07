from abc import ABC, abstractmethod
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


class Vector(ABC):
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

    @abstractmethod
    def similarity(self, other):
        raise NotImplementedError


class DenseVector(Vector):
    def similarity(self, other):
        return 1 - spatial.distance.cosine(self.value, other.value)


class SparseVector(Vector):
    def similarity(self, other):
        return cosine_similarity(self.value, other.value)


class Similarity(ABC):
    """
    Class for the various types of similarity
    """
    def __init__(self):
        pass

    @abstractmethod
    def perform(self, v1: Vector, v2: Vector):
        """
        Calculates the similarity between v1 and v2
        """
        raise NotImplementedError


class CosineSimilarity(Similarity):
    """
    Computes cosine similarity of given numpy arrays
    """
    def __init__(self):
        super().__init__()

    def perform(self, v1: Vector, v2: Vector):
        return v1.similarity(v2)

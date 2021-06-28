from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np


class Similarity(ABC):
    """
    Class for the various types of similarity
    """
    def __init__(self):
        pass

    @abstractmethod
    def perform(self, v1: np.ndarray, v2: np.ndarray):
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

    def perform(self, v1: np.ndarray, v2: np.ndarray):
        """
        Calculates the cosine similarity between v1 and v2
        """

        # Cosine_distance is defined in the scipy library as 1 - cosine_similarity, so:
        # 1 - cosine_distance = 1 - (1 - cosine_similarity) = cosine_similarity
        return 1 - cosine_distance(v1, v2)

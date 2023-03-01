from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class Similarity(ABC):
    """
    Abstract Class for the various types of similarity
    """
    def __init__(self):
        pass

    @abstractmethod
    def perform(self, v1: Union[np.ndarray, sparse.csr_matrix], v2: Union[np.ndarray, sparse.csr_matrix]):
        """
        Calculates the similarity between v1 and v2
        """
        raise NotImplementedError


class CosineSimilarity(Similarity):
    """
    Computes cosine similarity
    """
    def __init__(self):
        super().__init__()

    def perform(self, v1: Union[np.ndarray, sparse.csr_matrix], v2: Union[np.ndarray, sparse.csr_matrix]):
        """
        Calculates the cosine similarity between v1 and v2

        Args:
            v1: first numpy array
            v2: second numpy array
        """

        return cosine_similarity(v1, v2, dense_output=True)

    def __str__(self):
        return "CosineSimilarity"

    def __repr__(self):
        return f"CosineSimilarity()"

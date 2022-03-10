import numpy as np
from abc import ABC, abstractmethod


class CombiningTechnique(ABC):
    """
    Class that generalizes the modality in which loaded embeddings will be
    combined to produce a semantic representation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def combine(self, embedding_matrix: np.ndarray):
        """
        Combine, in a way specified in the implementations,
        the row of the input matrix

        Args:
            embedding_matrix: matrix whose rows will be combined

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        return f'CombiningTechnique'


class Centroid(CombiningTechnique):
    """
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the centroid vector of a matrix.
    """

    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array whose centroid will be calculated

        Returns:
            np.ndarray: centroid vector of the input matrix
        """
        return np.mean(embedding_matrix, axis=0)

    def __str__(self):
        return "Centroid"

    #@abstractmethod
    def __repr__(self):
        return f'Centroid'


class Sum(CombiningTechnique):
    """
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the sum vector of a matrix.
    """

    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array whose sum will be calculated

        Returns:
            np.ndarray: sum vector of the input matrix
        """
        return np.sum(embedding_matrix, axis=0)

    def __str__(self):
        return "Vector sum"

    @abstractmethod
    def __repr__(self):
        return f'Sum'


class SingleToken(CombiningTechnique):
    def __init__(self, token_index: int):
        self.token_index = token_index
        super().__init__()

    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        try:
            sentence_embedding = embedding_matrix[self.token_index]
        except IndexError:
            raise IndexError(f'The embedding matrix has {embedding_matrix.shape[1]} '
                             f'embeddings but you tried to take the {self.token_index + 1}th')
        return sentence_embedding

    @abstractmethod
    def __repr__(self):
        return f'SingleToken(token index= {self.token_index}'

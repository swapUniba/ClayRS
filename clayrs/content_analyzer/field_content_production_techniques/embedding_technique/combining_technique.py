import numpy as np
from abc import ABC, abstractmethod


class CombiningTechnique(ABC):
    """
    Class that generalizes the modality in which loaded embeddings will be
    combined to produce a semantic representation.
    """

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
        raise NotImplementedError


class Centroid(CombiningTechnique):
    """
    This class computes the centroid vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix: np bi-dimensional array where rows are words columns are hidden dimension
                whose centroid will be calculated

        Returns:
            Centroid vector of the input matrix
        """
        return np.nanmean(embedding_matrix, axis=0)

    def __str__(self):
        return "Centroid"

    def __repr__(self):
        return f'Centroid()'


class Sum(CombiningTechnique):
    """
    This class computes the sum vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates the sum vector of the input matrix

        Args:
            embedding_matrix: np bi-dimensional array where rows are words columns are hidden dimension
                whose sum vector will be calculated

        Returns:
            Sum vector of the input matrix
        """
        return np.sum(embedding_matrix, axis=0)

    def __str__(self):
        return "Sum"

    def __repr__(self):
        return f'Sum()'


class SingleToken(CombiningTechnique):
    """
    Class which takes a specific row as representative of the whole matrix

    Args:
        token_index: index of the row of the matrix to take
    """
    def __init__(self, token_index: int):
        self.token_index = token_index
        super().__init__()

    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Takes the row with index `token_index` (set in the constructor) from the input `embedding_matrix`

        Args:
            embedding_matrix: np bi-dimensional array where rows are words columns are hidden dimension
                from where the single token will be extracted

        Returns:
            Single row as representative of the whole matrix

        Raises:
            IndexError: Exception raised when `token_index` (set in the constructor) is out of bounds for the input
                matrix
        """
        try:
            sentence_embedding = embedding_matrix[self.token_index]
        except IndexError:
            raise IndexError(f'The embedding matrix has {embedding_matrix.shape[1]} '
                             f'embeddings but you tried to take the {self.token_index+1}th')
        return sentence_embedding

    def __str__(self):
        return "SingleToken"

    def __repr__(self):
        return f"SingleToken(token_index={self.token_index})"

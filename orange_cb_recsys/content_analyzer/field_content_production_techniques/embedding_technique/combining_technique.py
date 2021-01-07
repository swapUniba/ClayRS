import numpy as np

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    CombiningTechnique


class Centroid(CombiningTechnique):
    """"
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the centroid vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """"
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array whose centroid will be calculated

        Returns:
            np.ndarray: centroid vector of the input matrix
        """
        return np.average(embedding_matrix, axis=0)

    def __str__(self):
        return "Centroid"

    def __repr__(self):
        return "< Centroid >"


class Sum(CombiningTechnique):
    """"
    Class that implements the Abstract Class CombiningTechnique,
    this class implements the sum vector of a matrix.
    """
    def combine(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """"
        Calculates the centroid of the input matrix

        Args:
            embedding_matrix (np.ndarray): np bi-dimensional array whose sum will be calculated

        Returns:
            np.ndarray: sum vector of the input matrix
        """
        return np.sum(embedding_matrix, axis=0)

    def __str__(self):
        return "Vector sum"

    def __repr__(self):
        return "< Vector sum >"

# your combining technique

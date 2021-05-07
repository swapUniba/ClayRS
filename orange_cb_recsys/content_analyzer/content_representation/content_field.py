from abc import ABC, abstractmethod

from typing import Dict, Union
import numpy as np

from orange_cb_recsys.content_analyzer.content_representation.representation_container import RepresentationContainer


class FieldRepresentation(ABC):
    """
    Abstract class that generalizes the concept of "field representation",
    a field representation is a semantic way to represent a field of an item.
    """

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError


class FeaturesBagField(FieldRepresentation):
    """
    Class for field representation using a bag of features.
    This class can also be used to represent a bag of words: <keyword, score>;
    this representation is produced by the EntityLinking and tf-idf techniques

    Args:
        features (dict<str, object>): the dictionary where features are stored
    """

    def __init__(self, features: Dict[str, object] = None):
        super().__init__()
        if features is None:
            features = {}
        self.__features: Dict[str, object] = features

    @property
    def value(self) -> Dict[str, object]:
        """
        Get the features dict

        Returns:
            features (dict<str, object>): the features dict
        """
        return self.__features

    def __str__(self):
        return "\n %s" % str(self.__features)

    def __eq__(self, other):
        return self.__features == other.__features


class StringField(FieldRepresentation):
    """
    Class for field with no complex representation.

    Args:
        value (str): string representing the value of the field
    """

    def __init__(self, value: str = None):
        super().__init__()
        self.__value: str = value

    @property
    def value(self) -> str:
        return self.__value

    def __str__(self):
        return "\n %s" % str(self.__value)

    def __eq__(self, other):
        return self.__value == other.__value


class EmbeddingField(FieldRepresentation):
    """
    Class for field representation using embeddings (dense numeric vectors)
    this representation is produced by the EmbeddingTechnique.

    Examples:
        shape (4) = [x,x,x,x]
        shape (2,2) = [[x,x],
                       [x,x]]

    Args:
        embedding_array (np.ndarray): embeddings array,
            it can be of different shapes according to the granularity of the technique
    """

    def __init__(self, embedding_array: np.ndarray):
        super().__init__()
        self.__embedding_array: np.ndarray = embedding_array

    @property
    def value(self) -> np.ndarray:
        return self.__embedding_array

    def __str__(self):
        return " \n %s" % str(self.__embedding_array)

    def __eq__(self, other):
        return self.__embedding_array == other.__embedding_array


class ContentField:
    """
    Class that represents a field, a field can have more than one representation for itself that are identified
    by a specific representation name

    Args:
        representation_container (RepresentationContainer): object that stores the ids for the representations
        and the representation's instances in a dataframe
    """

    def __init__(self, representation_container: RepresentationContainer = None):
        if representation_container is None:
            representation_container = RepresentationContainer()

        self.__representation_container: RepresentationContainer = representation_container

    def get_representation(self, representation_id: Union[str, int]) -> FieldRepresentation:
        """
        Getter for the FieldRepresentation instance of a specific field representation name

        Returns:
            FieldRepresentation: instance of the representation for the name passed as argument
        """
        return self.__representation_container[representation_id]

    def append(self, representation: FieldRepresentation, representation_id: str = None):
        """
        Sets the field representation for a specific representation name

        Args:
            representation_id (str): name of the field representation
            representation (FieldRepresentation): field representation's content
        """
        self.__representation_container.append(representation, representation_id)

    def __eq__(self, other) -> bool:
        """
        override of the method __eq__ of object class,

        Args:
            other (ContentField): the field to check if is equal to self

        Returns:
            bool: True if the names are equals
        """
        return self.__representation_container == other.__representation_container

    def __str__(self):
        rep_string = '\n\n'.join("Representation: \n" + str(self.__representation_container))

        return "\n\n %s \n------------------------------------" % rep_string

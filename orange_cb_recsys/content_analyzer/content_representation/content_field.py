from abc import ABC, abstractmethod

from typing import Dict
import numpy as np


class FieldRepresentation(ABC):
    """
    Abstract class that generalizes the concept of "field representation",
    a field representation is a semantic way to represent a field of an item.

    Args:
        name (str): name of the representation's instance
    """

    def __init__(self, name: str):
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

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

    def __init__(self, name: str, features: Dict[str, object] = None):
        super().__init__(name)
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

    def append_feature(self, feature_key: str, feature_value):
        """
        Add a feature (feature_key, feature_value) to the dict

        Args:
            feature_key (str): key, can be a url or a keyword
            feature_value: the value of the field
        """
        self.__features[feature_key] = feature_value

    def get_feature(self, feature_key):
        """
        Get the feature_value from the dict[feature_key]

        Args:
            feature_key (str): key, can be a url or a keyword

        Returns:
            feature_value: the value of the field
        """
        return self.__features[feature_key]

    def __eq__(self, other):
        return self.__features == other.__features

    def __str__(self):
        representation_string = "Representation: " + self.name
        return "%s \n %s" % (representation_string, str(self.__features))


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

    def __init__(self, name: str,
                 embedding_array: np.ndarray):
        super().__init__(name)
        self.__embedding_array: np.ndarray = embedding_array

    @property
    def value(self) -> np.ndarray:
        return self.__embedding_array

    def __str__(self):
        representation_string = "Representation: " + self.name
        return "%s \n\n %s" % (representation_string, str(self.__embedding_array))

    def __eq__(self, other):
        return self.__embedding_array == other.__embedding_array


class ContentField:
    """
    Class that represents a field, a field can have more than one representation for itself

    Args:
        field_name (str): the name of the field
        timestamp (str): string that represents the timestamp
        representation_dict (dict<str, FieldRepresentation>): Dictionary whose keys are the name
            of the various representations, and the values are the corresponding FieldRepresentation
            instances.
    """

    def __init__(self, field_name: str,
                 timestamp: str = None,
                 representation_dict: Dict[str, FieldRepresentation] = None):
        if representation_dict is None:
            representation_dict = {}
        self.__timestamp = timestamp
        self.__field_name: str = field_name
        self.__representation_dict: Dict[str, object] = representation_dict

    @property
    def name(self) -> str:
        return self.__field_name

    def append(self, representation_id: str, representation: FieldRepresentation):
        self.__representation_dict[representation_id] = representation

    def get_representation(self, representation_id: str):
        return self.__representation_dict[representation_id]

    def __eq__(self, other) -> bool:
        """
        override of the method __eq__ of object class,

        Args:
            other (ContentField): the field to check if is equal to self

        Returns:
            bool: True if the names are equals
        """
        return self.__field_name == \
               other.name and self.__representation_dict == other.__representation_dict

    def __str__(self):
        field_string = "Field:" + self.__field_name
        rep_string = '\n\n'.join(str(rep) for rep in self.__representation_dict.values())

        return "%s \n\n %s ------------------------------------" % (field_string, rep_string)

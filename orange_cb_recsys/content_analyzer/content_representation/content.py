from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np

from orange_cb_recsys.content_analyzer.content_representation.representation_container import RepresentationContainer
from orange_cb_recsys.content_analyzer.memory_interfaces import InformationInterface


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
        return str(self.__features)

    def __eq__(self, other):
        return self.__features == other.__features


class SimpleField(FieldRepresentation):
    """
    Class for field with no complex representation.

    Args:
        value (str): string representing the value of the field
    """

    def __init__(self, value: object = None):
        super().__init__()
        self.__value: object = value

    @property
    def value(self) -> object:
        return self.__value

    def __str__(self):
        return str(self.__value)

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
        return str(self.__embedding_array)

    def __eq__(self, other):
        return self.__embedding_array == other.__embedding_array


class IndexField(FieldRepresentation):
    """
    Class for field representation using an index.
    Allows to dynamically retrieve the contents representations serialized in an index.

    Args:
        field_name (str): field's field_name located in the index
            N.B. : it might differ from the original field_name, for example "Plot" might be "Plot_0"
        index_id (int): position of the content in the index
        index (InformationInterface): index from which the data will be retrieved
    """

    def __init__(self, field_name: str, index_id: int, index: InformationInterface):
        super().__init__()
        self.__field_name = field_name
        self.__index_id = index_id
        self.__index = index

    @property
    def value(self) -> str:
        return self.__index.get_field(self.__field_name, self.__index_id)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.__field_name == other.__field_name \
               and self.__index_id == other.__content_id \
               and self.__index == other.__index


class ExogenousPropertiesRepresentation(ABC):
    """
    Output of LodPropertiesRetrieval, different representations exist according to different techniques
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError


class PropertiesDict(ExogenousPropertiesRepresentation):
    """
    Couples <property name, property value> retrieved by DBPediaMappingTechnique

    Args:
        features: properties in the specified format
    """

    def __init__(self, features: Dict[str, str] = None):
        super().__init__()
        if features is None:
            features = {}

        self.__features: Dict[str, str] = features

    @property
    def value(self):
        """
        Returns: features dictionary
        """
        return self.__features

    def __str__(self):
        return str(self.__features)


class Content:
    """
    Class that represents a content. A content can be an item or a user.
    A content is identified by a string id and is composed by different fields and different exogenous representations.

    The fields and the exogenous representations are stored as Representation Containers and they should not be
    available to the end user. Methods that return Representation Containers must be exclusively used for internal
    purposes.

    Args:
        content_id (str): string identifier
        field_dict (dict[str, RepresentationContainer]): dictionary containing the fields instances for the content,
            and their name as dictionary key
        exogenous_rep_container (RepresentationContainer): different representations of content obtained
            using ExogenousPropertiesRetrieval
    """
    def __init__(self, content_id: str,
                 field_dict: Dict[str, RepresentationContainer] = None,
                 exogenous_rep_container: RepresentationContainer = None):
        if field_dict is None:
            field_dict = {}       # list o dict
        if exogenous_rep_container is None:
            exogenous_rep_container = RepresentationContainer()

        self.__content_id: str = content_id
        self.__field_dict: Dict[str, RepresentationContainer] = field_dict
        self.__exogenous_rep_container: RepresentationContainer = exogenous_rep_container

    @property
    def content_id(self):
        """
        Getter for the content id
        """
        return self.__content_id

    @property
    def field_dict(self):
        """
        Getter for the dictionary containing the content's fields
        """
        return self.__field_dict

    @property
    def exogenous_rep_container(self):
        """
        Getter for the dictionary containing the content's exogenous representations
        """
        return self.__exogenous_rep_container

    def append_field(self, field_name: str, field: RepresentationContainer):
        """
        Sets a specific field for the content

        Args:
            field_name (str): field name to set
            field (ContentField): represents the data in the field and it will be set for the said field_name
        """
        self.__field_dict[field_name] = field

    def get_field(self, field_name: str) -> RepresentationContainer:
        """
        Getter for the ContentField of a specific field_name

        Args:
            field_name (str): field for which the ContentField will be retrieved
        """
        return self.__field_dict[field_name]

    def remove_field(self, field_name: str):
        """
        Removes the field named field_name from the field dictionary

        Args:
            field_name (str): the name of the field to remove
        """
        self.__field_dict.pop(field_name)

    def append_field_representation(self, field_name: str, representation: FieldRepresentation,
                                    representation_id: str = None):
        """
        Adds the given representation to the specific field_name with an external_id corresponding to the representation
        id (if passed). If the field name is not in the content, it will be added.

        Args:
            field_name (str): field_name to which the specific representation will be added
            representation (FieldRepresentation): field representation's content
            representation_id (str): name of the field representation
        """
        if field_name not in self.__field_dict.keys():
            self.__field_dict[field_name] = RepresentationContainer()
        self.__field_dict[field_name].append(representation, representation_id)

    def get_field_representation(self, field_name: str, representation_id: Union[int, str]) -> FieldRepresentation:
        """
        Getter for the FieldRepresentation instance of a specific field representation name

        Args:
            field_name (str): field_name from which the specific representation will be extracted
            representation_id (Union[int, str]): id of the specific representation (either the internal or external id)

        Returns:
            FieldRepresentation: instance of the representation for the name passed as argument
        """
        return self.__field_dict[field_name][representation_id]

    def remove_field_representation(self, field_name: str, representation_id: Union[int, str]):
        """
        Removes a specific representation, identified by the id passed as argument (either internal or external id),
        from a specific field, identified by the field name passed as argument

        Args:
            field_name (str): field_name from which the specific representation will be removed
            representation_id (Union[int, str]): id of the specific representation (either the internal or external id)
        """
        self.__field_dict[field_name].pop(representation_id)

    def append_exogenous(self, exogenous_properties: ExogenousPropertiesRepresentation, exo_name: str = None):
        """
        Sets a specific exogenous representation for the content. If an exo_name is defined, it will be added
        as external_id to the representation

        Args:
            exo_name (str): name of the representation, it can be used to refer to the representation
            exogenous_properties (ExogenousPropertiesRepresentation): represents the data in the
                exogenous representation
        """
        self.__exogenous_rep_container.append(exogenous_properties, exo_name)

    def get_exogenous(self, exo_name: Union[int, str]) -> ExogenousPropertiesRepresentation:
        """
        Getter for the ExogenousPropertiesRepresentation of a specific exogenous representation name

        Args:
            exo_name (str): representation's name for which the ExogenousPropertiesRepresentation will be retrieved
        """
        return self.__exogenous_rep_container[exo_name]

    def remove_exogenous(self, exo_name: Union[str, int]):
        """
        Removes the exogenous representation named exo_name from the exogenous representation container

        Args:
            exo_name (str): the name of the exogenous representation to remove
        """
        self.__exogenous_rep_container.pop(exo_name)

    def __hash__(self):
        return hash(self.__content_id)

    def __repr__(self):
        return str(self.__content_id)

    def __str__(self):
        content_string = "Content: %s\n" % self.__content_id
        exo_string = "Exogenous representations:\n"
        if len(self.__exogenous_rep_container) != 0:
            exo_string += "%s\n" % self.__exogenous_rep_container
        else:
            exo_string = "\nNo representation found for the Content!\n"

        if len(self.__field_dict.items()) != 0:
            field_string = ''
            for field, rep in self.__field_dict.items():
                field_string += "\nField: %s \n%s\n" % (field, rep)
        else:
            field_string = "Field representations:\n\nNo representation found for the Content fields!"

        return "%s\n%s\n%s\n##############################" % (content_string, exo_string, field_string)

    def __eq__(self, other):
        return self.__content_id == other.__content_id and self.__field_dict == other.__field_dict

from abc import ABC, abstractmethod
from typing import Dict

from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField


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


class Content:
    """
    Class that represents a content. A content can be an item or a user.
    A content is identified by a string id and is composed by different fields and different exogenous representations.

    Args:
        content_id (str): string identifier
        field_dict (dict[str, ContentField]): dictionary containing the fields instances for the content,
            and their name as dictionary key
        exogenous_rep_dict (Dict <str, ExogenousProperties>): different representations of content obtained
            using ExogenousPropertiesRetrieval, the dictionary key is the representation name
    """
    def __init__(self, content_id: str,
                 field_dict: Dict[str, ContentField] = None,
                 exogenous_rep_dict: Dict[str, ExogenousPropertiesRepresentation] = None):
        if field_dict is None:
            field_dict = {}       # list o dict
        if exogenous_rep_dict is None:
            exogenous_rep_dict = {}

        self.__content_id: str = content_id
        self.__field_dict: Dict[str, ContentField] = field_dict
        self.__exogenous_rep_dict: Dict[str, ExogenousPropertiesRepresentation] = exogenous_rep_dict
        self.__index_document_id: int = None  # to be removed

    @property
    def content_id(self):
        """
        Getter for the content id
        """
        return self.__content_id

    @property
    def index_document_id(self) -> int:
        return self.__index_document_id

    @index_document_id.setter
    def index_document_id(self, index_document_id: int):
        self.__index_document_id = index_document_id

    @property
    def field_dict(self):
        """
        Getter for the dictionary containing the content's fields
        """
        return self.__field_dict

    @property
    def exogenous_rep_dict(self):
        """
        Getter for the dictionary containing the content's exogenous representations
        """
        return self.__exogenous_rep_dict

    def append_field(self, field_name: str, field: ContentField):
        """
        Sets a specific field for the content

        Args:
            field_name (str): field name to set
            field (ContentField): represents the data in the field and it will be set for the said field_name
        """
        self.__field_dict[field_name] = field

    def get_field(self, field_name: str) -> ContentField:
        """
        Getter for the ContentField of a specific field_name

        Args:
            field_name (str): field for which the ContentField will be retrieved
        """
        return self.__field_dict[field_name]

    def append_exogenous(self, exo_name: str, exogenous_properties: ExogenousPropertiesRepresentation):
        """
        Sets a specific exogenous representation for the content

        Args:
            exo_name (str): name of the representation, it can be used to refer to the representation
            exogenous_properties (ExogenousPropertiesRepresentation): represents the data in the
                exogenous representation
        """
        self.__exogenous_rep_dict[exo_name] = exogenous_properties

    def get_exogenous(self, exo_name: str) -> ExogenousPropertiesRepresentation:
        """
        Getter for the ExogenousPropertiesRepresentation of a specific exogenous representation name

        Args:
            exo_name (str): representation's name for which the ExogenousPropertiesRepresentation will be retrieved
        """
        return self.__exogenous_rep_dict[exo_name]

    def remove_field(self, field_name: str):
        """
        Removes the field named field_name from the field dictionary

        Args:
            field_name (str): the name of the field to remove
        """
        self.__field_dict.pop(field_name)

    def remove_exogenous(self, exo_name: str):
        """
        Removes the exogenous representation named exo_name from the exogenous representation dictionary

        Args:
            exo_name (str): the name of the exogenous representation to remove
        """
        self.__exogenous_rep_dict.pop(exo_name)

    def __str__(self):
        content_string = "Content: %s" % self.__content_id
        field_string = ''
        for field, rep in self.__field_dict.items():
            field_string += "\nField: %s %s" % (field, rep)

        return "%s \n\n %s \n##############################" % (content_string, field_string)

    def __eq__(self, other):
        return self.__content_id == other.__content_id and self.__field_dict == other.__field_dict

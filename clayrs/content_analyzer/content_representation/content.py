from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Union, List, Tuple, TYPE_CHECKING
import numpy as np
import json
from numbers import Number

from scipy import sparse

if TYPE_CHECKING:
    from clayrs.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface

from clayrs.content_analyzer.content_representation.representation_container import RepresentationContainer


class FieldRepresentation(ABC):
    """
    Abstract class that generalizes the concept of "field representation",
    a field representation is a semantic way to represent a field of an item.
    """
    __slots__ = ()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    def to_json(self):
        """
        The Json representation of each complex representation it's just its string representation, but there might be
        the case in which the representation is more complex than that
        """
        return str(self)


class FeaturesBagField(FieldRepresentation):
    """
    Class for field representation using a bag of features.
    This class can also be used to represent a bag of words: <keyword, score>;
    this representation is produced by the EntityLinking and tf-idf techniques

    Args:
        sparse_scores: the sparse matrix where features are stored
    """
    __slots__ = ('__scores', '__pos_feature_tuples')

    def __init__(self, sparse_scores: sparse.csc_matrix, pos_feature_tuples: List[Tuple[int, str]]):
        self.__scores = sparse_scores
        self.__pos_feature_tuples = pos_feature_tuples

    @property
    def value(self) -> sparse.csc_matrix:
        """
        Get the features dict

        Returns:
            features (dict<str, object>): the features dict
        """
        return self.__scores

    def to_json(self):
        tuple_representation = np.array([(coordinates_tuple, self.value[coordinates_tuple])
                                         for coordinates_tuple in zip(*self.value.nonzero())], dtype=object)

        return dict(sparse_tfidf=np.array2string(tuple_representation, threshold=np.inf, separator=','),
                    pos_word_tuples=str(self.__pos_feature_tuples),
                    len_vocabulary=self.__scores.shape[1])

    def __str__(self):
        return str(self.__scores)

    def __eq__(self, other):
        return np.array_equal(self.__scores, other.__scores) and self.__pos_feature_tuples == other.__pos_feature_tuples


class SimpleField(FieldRepresentation):
    """
    Class for field with no complex representation.

    Args:
        value (str): string representing the value of the field
    """
    __slots__ = ('__value',)

    def __init__(self, value: object = None):
        self.__value: object = value

    @property
    def value(self) -> object:
        return self.__value

    def __str__(self):
        return str(self.__value)

    def __repr__(self):
        return '{} - {}'.format(str(self), type(self.value))

    def __eq__(self, other):
        return self.__value == other.__value


class EmbeddingField(FieldRepresentation, np.lib.mixins.NDArrayOperatorsMixin):
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
    __slots__ = ('__dense_array',)

    def __init__(self, embedding_array: np.ndarray):
        self.__dense_array = embedding_array

    @property
    def value(self) -> np.ndarray:
        return self.__dense_array

    def __str__(self):
        return np.array2string(self.__dense_array, threshold=np.inf, separator=',')

    def __eq__(self, other):
        return self.__dense_array == other.__dense_array

    def __array__(self, dtype=None):
        return self.__dense_array.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            scalars = []
            for input in inputs:
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    scalars.append(input.value)
                else:
                    return NotImplemented
            return self.__class__(ufunc(*scalars, **kwargs))
        else:
            return NotImplemented


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
    __slots__ = ('__field_name', '__index_id', '__index')

    def __init__(self, field_name: str, index_id: int, index: InformationInterface):
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
    __slots__ = ()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    def to_json(self):
        """
        The Json representation of each complex representation it's just its string representation, but there might be
        the case in which the representation is more complex than that
        """
        return str(self)


class PropertiesDict(ExogenousPropertiesRepresentation):
    """
    Couples <property name, property value> retrieved by DBPediaMappingTechnique

    Args:
        features: properties in the specified format
    """
    __slots__ = ('__features',)

    def __init__(self, features: Dict[str, str] = None):
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


class EntitiesProp(ExogenousPropertiesRepresentation):
    """
    Couples <property name, property value> retrieved by DBPediaMappingTechnique

    Args:
        features: properties in the specified format
    """
    __slots__ = ('__features',)

    def __init__(self, features: Dict[str, Dict] = None):
        if features is None:
            features = {}

        self.__features: Dict[str, Dict] = features

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
    __slots__ = ('__content_id', '__field_dict', '__exogenous_rep_container')

    def __init__(self, content_id: str,
                 field_dict: Dict[str, RepresentationContainer] = None,
                 exogenous_rep_container: RepresentationContainer = None):
        if field_dict is None:
            field_dict = {}  # list o dict
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

    def append_field_representation(self, field_name: str,
                                    representation: Union[List[FieldRepresentation], FieldRepresentation],
                                    representation_id: Union[List[str], str] = None):
        """
        Adds the given representation(s) to the specific field_name with an external_id corresponding to
        the representation id(s) (if passed). If the field name is not in the content, it will be added.

        It can be passed a single representation and a single id, or a list of representations and a corresponding
        list of ids. In this case the two lists must be of the same length and ids are linked to representations
        by position.
        EXAMPLE:
            representation = [rep1, rep2, rep3]
            representation_id = [id_rep1, id_rep2, id_rep3]

        Args:
            field_name (str): field_name to which the specific representation will be added
            representation (Union[List[FieldRepresentation], FieldRepresentation]): single representation or a list of
                representations for the content
            representation_id (Union[List[str], str]): single name of the field representation or a list of names for
                the field representations.
        """
        if isinstance(representation, list) and representation_id is None:
            representation_id = [None for _ in range(len(representation))]

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

    def append_exogenous_representation(self, exogenous_properties: Union[List[ExogenousPropertiesRepresentation],
                                                                          ExogenousPropertiesRepresentation],
                                        exo_name: Union[List[str], str] = None):
        """
        Sets a specific or multiple exogenous representations for the content. If an exo_name or a list of exo name is
        defined, it will be added as external_id to the representation(s)

        It can be passed a single representation and a single exo_name, or a list of representations and a corresponding
        list of exo_names. In this case the two lists must be of the same length and exo_names are linked to
        representations by position.
        EXAMPLE:
            exogenous_properties = [rep1, rep2, rep3]
           exo_name = [exo_name_rep1, exo_name_rep2, exo_name_rep3]

        Args:
            exogenous_properties (Union[List[ExogenousPropertiesRepresentation], ExogenousPropertiesRepresentation]):
                single exogenous representation or a list of exogenous representations for the content
            exo_name (Union[List[str], str]): single name of the exogenous representation or a list of names for
                the exogenous representations.
        """
        if isinstance(exogenous_properties, list) and exo_name is None:
            exo_name = [None for _ in range(len(exogenous_properties))]

        self.__exogenous_rep_container.append(exogenous_properties, exo_name)

    def get_exogenous_representation(self, exo_name: Union[int, str]) -> ExogenousPropertiesRepresentation:
        """
        Getter for the ExogenousPropertiesRepresentation of a specific exogenous representation name

        Args:
            exo_name (str): representation's name for which the ExogenousPropertiesRepresentation will be retrieved
        """
        return self.__exogenous_rep_container[exo_name]

    def remove_exogenous_representation(self, exo_name: Union[str, int]):
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
            exo_string += "\nNo representation found for the Content!\n"

        if len(self.__field_dict.items()) != 0:
            field_string = ''
            for field, rep in self.__field_dict.items():
                field_string += "\nField: %s \n%s\n" % (field, rep)
        else:
            field_string = "Field representations:\n\nNo representation found for the Content fields!"

        return "%s\n%s\n%s\n##############################" % (content_string, exo_string, field_string)

    def __eq__(self, other):
        result = False
        try:
            result = self.__content_id == other.__content_id and self.__field_dict == other.__field_dict
        except AttributeError:
            pass

        return result


class ContentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Content):
            content = {'content_id': obj.content_id}
            for row in obj.exogenous_rep_container:
                key = f"Exo#{row['internal_id']}"
                # polymorphic call to 'to_json()' function
                content[key] = row['representation'].to_json()

            for field in obj.field_dict:
                field_container = obj.field_dict[field]

                for row in field_container:
                    key = f"{field}#{row['internal_id']}"
                    # polymorphic call to 'to_json()' function
                    content[key] = row['representation'].to_json()

            return content
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

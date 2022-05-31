from abc import ABC, abstractmethod
from typing import Union
import shutil


class InformationInterface(ABC):
    """
    Abstract class that deals with the serialization
    and deserialization of a field (of a content) data
    basing on the type of element extracted.

    Args:
        directory (str): directory where to store the serialized content and where to access for deserialization
    """

    def __init__(self, directory: str):
        self.__directory: str = directory

    def delete(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    @property
    def directory(self):
        return self.__directory

    @abstractmethod
    def new_content(self):
        """
        Creates a new item, that will be serialized by the apposite method.
        """
        raise NotImplementedError

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        """
        Serialize the raw data of a field.

        Args:
            field_name: name of the created field
            field_data: data to serialize
        """
        raise NotImplementedError

    @abstractmethod
    def serialize_content(self):
        """
        Add to the serialized collection the current item
        """
        raise NotImplementedError

    @abstractmethod
    def init_writing(self, delete_old: bool = False):
        """
        Set the interface in writing mode,
        tf the specified directory does not exit a new one will be created.
        If the directory exists and an object associated to the IndexInterface already exists, what happens depends on
        the delete_old attribute

        Args:
            delete_old (bool): if True, the object handled by the information interface that was in the same directory
                is destroyed and replaced; if False, the object is simply opened
        """
        raise NotImplementedError

    @abstractmethod
    def stop_writing(self):
        """
        Stop writing mode
        """
        raise NotImplementedError

    @abstractmethod
    def get_field(self, field_name: str, content_id: Union[str, int]):
        """
        Allows to retrieve the content stored in a field for a content

        Args:
            field_name (str): name of the field from which the data will be retrieved
            content_id (Union[str, int]): either the position or Id of the content that contains the specified field
        """
        raise NotImplementedError

    def __hash__(self):
        return hash(str(self.__directory))

    def __eq__(self, other):
        if isinstance(other, InformationInterface):
            return self.directory == other.directory


class ImageInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in image format.
    """

    @abstractmethod
    def new_content(self):
        raise NotImplementedError

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        raise NotImplementedError

    @abstractmethod
    def serialize_content(self):
        raise NotImplementedError

    @abstractmethod
    def init_writing(self, delete_old: bool = False):
        raise NotImplementedError

    @abstractmethod
    def stop_writing(self):
        raise NotImplementedError

    @abstractmethod
    def get_field(self, field_name: str, content_id: Union[str, int]):
        raise NotImplementedError


class AudioInterface(InformationInterface):
    """
    Future feature
    Abstract class to use when the field information is in audio format.
    """

    @abstractmethod
    def new_content(self):
        raise NotImplementedError

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        raise NotImplementedError

    @abstractmethod
    def serialize_content(self):
        raise NotImplementedError

    @abstractmethod
    def init_writing(self, delete_old: bool = False):
        raise NotImplementedError

    @abstractmethod
    def stop_writing(self):
        raise NotImplementedError

    @abstractmethod
    def get_field(self, field_name: str, content_id: Union[str, int]):
        raise NotImplementedError


class TextInterface(InformationInterface):
    """
    Abstract class to use when the field information is textual.
    """

    @abstractmethod
    def new_content(self):
        raise NotImplementedError

    @abstractmethod
    def new_field(self, field_name: str, field_data):
        raise NotImplementedError

    @abstractmethod
    def serialize_content(self):
        raise NotImplementedError

    @abstractmethod
    def init_writing(self, delete_old: bool = False):
        raise NotImplementedError

    @abstractmethod
    def stop_writing(self):
        raise NotImplementedError

    @abstractmethod
    def get_field(self, field_name: str, content_id: Union[str, int]):
        raise NotImplementedError

    @abstractmethod
    def query(self, string_query: str, results_number: int, mask_list: list = None,
              candidate_list: list = None, classic_similarity: bool = True) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_tf_idf(self, field_name: str, content_id: Union[str, int]):
        raise NotImplementedError

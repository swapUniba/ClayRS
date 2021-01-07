from abc import ABC, abstractmethod
from typing import List


class InformationProcessor(ABC):
    """
    Abstract class that generalizes data processing.
    """
    def __init__(self):
        self.__lang = "EN"

    @property
    def lang(self):
        return self.__lang

    @lang.setter
    def lang(self, lang: str):
        self.__lang = lang

    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class ImageProcessor(InformationProcessor):
    """
    Abstract class for image processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class AudioProcessor(InformationProcessor):
    """
    Abstract class for audio processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class TextProcessor(InformationProcessor):
    """
    Abstract class for raw text processing.
    """
    @abstractmethod
    def process(self, field_data):
        raise NotImplementedError


class NLP(TextProcessor):
    """
    Class for processing a text via Natural Language Processing.

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        stemming (bool): Whether you want to perform stemming
        lemmatization (bool): Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
        named_entity_recognition (bool): Whether you want to perform named entity recognition
    """

    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 named_entity_recognition: bool = False):
        super().__init__()
        self.__stopwords_removal: bool = stopwords_removal
        self.__stemming: bool = stemming
        self.__lemmatization: bool = lemmatization
        self.__strip_multiple_whitespaces: bool = strip_multiple_whitespaces
        self.__url_tagging: bool = url_tagging
        self.__named_entity_recognition: bool = named_entity_recognition

    @property
    def stopwords_removal(self):
        return self.__stopwords_removal

    @property
    def stemming(self):
        return self.__stemming

    @property
    def lemmatization(self):
        return self.__lemmatization

    @property
    def strip_multiple_whitespaces(self):
        return self.__strip_multiple_whitespaces

    @property
    def url_tagging(self):
        return self.__url_tagging

    @property
    def named_entity_recognition(self):
        return self.__named_entity_recognition

    @stopwords_removal.setter
    def stopwords_removal(self, stopwords_removal):
        self.__stopwords_removal = stopwords_removal

    @stemming.setter
    def stemming(self, stemming):
        self.__stemming = stemming

    @lemmatization.setter
    def lemmatization(self, lemmatization):
        self.__lemmatization = lemmatization

    @strip_multiple_whitespaces.setter
    def strip_multiple_whitespaces(self, strip_multiple_whitespaces):
        self.__strip_multiple_whitespaces = strip_multiple_whitespaces

    @url_tagging.setter
    def url_tagging(self, url_tagging):
        self.__url_tagging = url_tagging

    @named_entity_recognition.setter
    def named_entity_recognition(self, named_entity_recognition: bool):
        self.__named_entity_recognition = named_entity_recognition

    @abstractmethod
    def process(self, field_data) -> List[str]:
        """
        Apply on the original text the required preprocessing steps
        Args:
            field_data: text on which NLP with specified phases will be applied

        Returns:
            list<str>: The text, after being processed with the specified NLP pipeline,
                is splitted in single words that are put into a list. The splitting is executed
                even if none of the preprocessing steps is computed.
        """
        raise NotImplementedError

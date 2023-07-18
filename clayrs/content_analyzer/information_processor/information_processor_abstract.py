from abc import ABC, abstractmethod
from typing import List, Any

import torch


class InformationProcessor(ABC):
    """
    Abstract class that generalizes data processing.
    """

    @abstractmethod
    def process(self, field_data: Any):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class ImageProcessor(InformationProcessor, torch.nn.Module):
    """
    Abstract class for image processing.
    """
    @abstractmethod
    def forward(self, field_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def process(self, field_data: torch.Tensor) -> torch.Tensor:
        return self.forward(field_data)

    def __eq__(self, other):
        return torch.nn.Module.__eq__(self, other)


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

    @staticmethod
    def list_to_string(text: List[str]) -> str:
        """
        Convert list of str in str
        Args: text (str): list of str
        Returns: str sentence
        """
        string_text = ' '.join([str(elem) for elem in text])
        return string_text

    @staticmethod
    def string_to_list(text: str) -> List[str]:
        """
        Covert str in list of str
        Args:
            text (str): str sentence

        Returns List <str>: List of words
        """
        list_text = list(text.split(" "))
        return list_text

    @abstractmethod
    def process(self, field_data: str):
        raise NotImplementedError


class NLP(TextProcessor):
    """
    Class for processing a text via Natural Language Processing.

    """

    @abstractmethod
    def process(self, field_data: str) -> List[str]:
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

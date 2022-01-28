import string

from typing import List

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.spellcorrect import SpellCorrector


class Ekphrasis:
    """
    Class to manage text to locate dates, currencies, etc.,
    unpack hashtags and correct spelling.

    """

    def __init__(self,
                 spell_check: bool = False,
                 unpack_hashtag: bool = False,
                 convert_words: List = [],
                 ):

        self.spell_check = spell_check
        self.unpack_hashtag = unpack_hashtag
        self.convert_words = convert_words

        if isinstance(spell_check, str):
            spell_check = spell_check.lower() == 'true'

        if isinstance(unpack_hashtag, str):
            unpack_hashtag = unpack_hashtag.lower() == 'true'

    @staticmethod
    def __list_to_string(text: List[str]) -> str:
        """
        Covert list of str in str
        Args:
            text (str): list of str

        Returns: str sentence

        """
        string_text = ' '.join([str(elem) for elem in text])
        return string_text

    @staticmethod
    def __string_to_list(text) -> List[str]:
        """
                Covert str in list of str
                Args:
                    text (str): str sentence

                Returns List <str>: List of words

        """
        list_text = list(text.split(" "))
        return list_text

    def __spell_check(self, field_data):
        """
        Correct any spelling errors
        Args:
            field_data: text to correct

        Returns:
            field_data: correct text

        """
        field_data = self.__string_to_list(field_data)
        sp = SpellCorrector()
        correct_sentence = []
        for word in field_data:
            correct_sentence.append(sp.correct(word))
        field_data = self.__list_to_string(correct_sentence)
        return field_data

    @staticmethod
    def __remove_punctuation(text) -> str:
        """
        Punctuation removal
        Args:
            text(str) : text with punct
        Returns:
            text_without_punct (str): text without punct
        """

        text_without_punct=text.translate(str.maketrans('', '', string.punctuation))
        return text_without_punct

    def __check_if_string(self, text) -> str:
        """
        Check if text is list of str or str
        Args:
            text

        Returns:
            text (str): str sentence
        """
        if isinstance(text, List):
            text=self.__list_to_string(text)
        return text

    def process(self, field_data)->List[str]:

        """

        Args:
            field_data: content to be processed

        Returns:
            field_data (List<str>): list of str or dict in case of named entity recognition

        """
        field_data=self.__check_if_string(field_data)

        text_processor = TextPreProcessor(normalize=self.convert_words,
                                          unpack_hashtags=self.unpack_hashtag)
        field_data = text_processor.pre_process_doc(field_data)
        if self.spell_check:

            field_data=self.__spell_check(field_data)
        field_data=self.__remove_punctuation(field_data)
        field_data=self.__string_to_list(field_data)
        return field_data

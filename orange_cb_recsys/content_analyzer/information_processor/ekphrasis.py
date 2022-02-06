import string
from typing import List

from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from ekphrasis.classes.segmenter import Segmenter

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
                 annotate_words: List = [],
                 unpack_contractions: bool = False,
                 annotate_emoji: bool = False,
                 word_segmenter: bool = False,
                 remove_punctuation: bool = False,
                 ):
        self.unpack_contractions = unpack_contractions
        self.spell_check = spell_check
        self.unpack_hashtag = unpack_hashtag
        self.convert_words = convert_words
        self.annotate_words = annotate_words
        self.annotate_emoji = annotate_emoji
        self.word_segmenter = word_segmenter
        self.remove_punctuation = remove_punctuation

        if isinstance(unpack_contractions, str):
            unpack_contractions = unpack_contractions.lower() == 'true'
        if isinstance(spell_check, str):
            spell_check = spell_check.lower() == 'true'
        if isinstance(unpack_hashtag, str):
            unpack_hashtag = unpack_hashtag.lower() == 'true'
        if isinstance(annotate_emoji, str):
            annotate_emoji = annotate_emoji.lower() == 'true'
        if isinstance(word_segmenter, str):
            word_segmenter = word_segmenter.lower() == 'true'
        if isinstance(remove_punctuation, str):
            remove_punctuation = remove_punctuation.lower() == 'true'

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
        sp = SpellCorrector()
        correct_sentence = []
        for word in field_data:
            if word[0] != "<" and word[0] != "#":
                if word in string.punctuation:
                    correct_sentence.append(word)
                else:
                    correct_sentence.append(sp.correct(word))
            else:
                correct_sentence.append(word)
        return correct_sentence

    def __remove_punctuation(self, text) -> List[str]:
        """
        Punctuation removal
        Args:
            text(str) : text with punct
        Returns:
            text_without_punct (str): text without punct
        """
        if isinstance(text, List):
            text=self.__list_to_string(text)
        text = re.sub(r"[^\w\d<>\s]+", '', text)
        return text

    def __check_if_string(self, text) -> str:
        """
        Check if text is list of str or str
        Args:
            text

        Returns:
            text (str): str sentence
        """
        if isinstance(text, List):
            text = self.__list_to_string(text)
        return text

    def __annotate_emoji(self):
        """

        Returns: Dict of emoticons

        """
        if self.annotate_emoji:
            return [emoticons]

    def __word_segmenter(self, field_data) -> List[str]:
        """
        Split words together
        Args:
            field_data: Text to be processed

        Returns (List[str): Text with splitted words

        """
        word_seg_list = []
        seg_eng = Segmenter(corpus="english")
        for w in field_data:
            word_seg_list.append((seg_eng.segment(w)))
        word_seg_list = self.__list_to_string(word_seg_list)
        return word_seg_list

    @staticmethod
    def __strip_multiple_whitespaces_operation(text) -> str:
        """
        Remove multiple whitespaces and '<repeated> marker on input text

        Args:
            text (str):

        Returns:
            str: input text, multiple whitespaces removed
        """
        text = re.sub('<repeated>', ' ', text)
        text = re.sub(' +', ' ', text)
        return text
    def process(self, field_data) -> List[str]:

        """

        Args:
            field_data: content to be processed

        Returns:
            field_data (List<str>): list of str or dict in case of named entity recognition

        """
        field_data = self.__check_if_string(field_data)

        text_processor = TextPreProcessor(tokenizer=SocialTokenizer(lowercase=False).tokenize,
                                          normalize=self.convert_words,
                                          unpack_hashtags=self.unpack_hashtag, annotate=self.annotate_words,
                                          unpack_contractions=self.unpack_contractions, dicts=self.__annotate_emoji())
        field_data = text_processor.pre_process_doc(field_data)
        if self.remove_punctuation:
            field_data = self.__remove_punctuation(field_data)
            field_data=self.__strip_multiple_whitespaces_operation(field_data)
        if self.spell_check:
            field_data = self.__spell_check(field_data)
        if self.word_segmenter:
            field_data = self.__word_segmenter(field_data)
        if isinstance(field_data, str):
            field_data = self.__string_to_list(field_data)
        return field_data



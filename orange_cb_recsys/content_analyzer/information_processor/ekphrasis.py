from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP

import string
from typing import List, Dict

from ekphrasis.classes.tokenizer import SocialTokenizer, Tokenizer
from ekphrasis.dicts.emoticons import emoticons
import re
from ekphrasis.classes.segmenter import Segmenter

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.spellcorrect import SpellCorrector


class Ekphrasis(NLP):
    """
    Class to manage text to locate dates, currencies, etc.,
    unpack hashtags and correct spelling.
    """

    def __init__(self,
                 omit: List = None,
                 normalize: List = None,
                 unpack_contractions: bool = False,
                 unpack_hashtags: bool = False,
                 annotate: List = None,
                 corrector: str = None,
                 tokenizer: Tokenizer = SocialTokenizer(lowercase=True).tokenize,
                 segmenter: str = None,
                 all_caps_tag: str = None,
                 spell_correct_elong: bool = False,
                 fix_text: bool = False,
                 additional_substitution: List[Dict] = None
                 ):

        self.text_processor = TextPreProcessor(tokenizer=tokenizer,
                                               corrector=corrector, omit=omit,
                                               normalize=normalize, segmenter=segmenter,
                                               unpack_hashtags=unpack_hashtags,
                                               annotate=annotate,
                                               all_caps_tag=all_caps_tag, unpack_contractions=unpack_contractions,
                                               spell_correct_elong=spell_correct_elong,
                                               fix_text=fix_text,
                                               dicts=additional_substitution
                                               )
        if corrector is not None:
            self.spell_check = SpellCorrector(corpus=corrector)

        if segmenter is not None:
            self.word_segmenter = Segmenter(corpus=segmenter)

    def __spell_check(self, field_data):
        """
        Correct any spelling errors
        Args:
            field_data: text to correct
        Returns:
            field_data: correct text
        """

        correct_sentence = []
        for word in field_data:
            if word[0] != "<" and word[0] != "#":
                if word in string.punctuation:
                    correct_sentence.append(word)
                else:
                    correct_sentence.append(self.spell_check.correct(word))
            else:
                correct_sentence.append(word)
        return correct_sentence

    def __word_segmenter(self, field_data) -> List[str]:
        """
        Split words together
        Args:
            field_data: Text to be processed
        Returns (List[str): Text with splitted words
        """
        word_seg_list = []
        for word in field_data:
            word_seg_list.append((self.word_segmenter.segment(word)))
        return word_seg_list

    def process(self, field_data) -> List[str]:

        """
        Args:
            field_data: content to be processed
        Returns:
            field_data (List<str>): list of str or dict in case of named entity recognition
        """
        field_data = self.text_processor.pre_process_doc(field_data)
        if self.spell_check:
            field_data=self.__spell_check(field_data)
        if self.word_segmenter:
            field_data = self.__word_segmenter(field_data)
        return field_data

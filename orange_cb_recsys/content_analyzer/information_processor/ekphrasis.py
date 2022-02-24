import itertools

from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP

from typing import List, Dict, Callable

from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.segmenter import Segmenter

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.spellcorrect import SpellCorrector


class Ekphrasis(NLP):
    """
    Class to manage text to locate dates, currencies, etc.,
    unpack hashtags and correct spelling.
    """

    def __init__(self, *,
                 omit: List = None,
                 normalize: List = None,
                 unpack_contractions: bool = False,
                 unpack_hashtags: bool = False,
                 annotate: List = None,
                 corrector: str = None,
                 tokenizer: Callable = SocialTokenizer(lowercase=True).tokenize,
                 segmenter: str = None,
                 all_caps_tag: str = None,
                 spell_correction: bool = False,
                 segmentation: bool = False,
                 dicts: List[Dict] = None,
                 spell_correct_elong: bool = False
                 ):
        """
        omit (list): choose what tokens that you want to omit from the text.
            possible values: ['email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'hashtag']
            Important Notes:
                        1 - put url at front, if you plan to use it.
                            Messes with the regexes!
                        2 - if you use hashtag then unpack_hashtags will
                            automatically be set to False

        normalize (list): choose what tokens that you want to normalize
            from the text.
            possible values: ['email', 'percent', 'money', 'phone', 'user',
                'time', 'url', 'date', 'hashtag']
            for example: myaddress@mysite.com will be transformed to <email>
            Important Notes:
                        1 - put url at front, if you plan to use it.
                            Messes with the regexes!
                        2 - if you use hashtag then unpack_hashtags will
                            automatically be set to False

        unpack_contractions (bool): Replace *English* contractions in
            ``text`` str with their unshortened forms
            for example: can't -> can not, wouldn't -> would not, and so on...

        unpack_hashtags (bool): split a hashtag to it's constituent words.
            for example: #ilikedogs -> i like dogs

        annotate (list): add special tags to special tokens.
            possible values: ['hashtag', 'allcaps', 'elongated', 'repeated']
            for example: myaddress@mysite.com -> myaddress@mysite.com <email>

        tokenizer (callable): callable function that accepts a string and
            returns a list of strings if no tokenizer is provided then
            the text will be tokenized on whitespace

        segmenter (str): define the statistics of what corpus you would
            like to use [english, twitter]

        corrector (str): define the statistics of what corpus you would
            like to use [english, twitter]

        all_caps_tag (str): how to wrap the capitalized words
            values [single, wrap, every]
            Note: applicable only when `allcaps` is included in annotate[]
                - single: add a tag after the last capitalized word
                - wrap: wrap all words with opening and closing tags
                - every: add a tag after each word

        spell_correct_elong (bool): choose if you want to perform
            spell correction after the normalization of elongated words.
            * significantly affects performance (speed)

        spell_correction (bool): choose if you want to perform
            spell correction to the text
            * significantly affects performance (speed)
        """

        # ekphrasis has default values for arguments not passed. So if they are not evaluated in our class,
        # we simply don't pass them to ekphrasis
        kwargs_to_pass = {argument: arg_value for argument, arg_value in zip(locals().keys(), locals().values())
                          if argument != 'self' and arg_value is not None}

        self.text_processor = TextPreProcessor(**kwargs_to_pass)

        self.spell_correct_elong = spell_correct_elong
        self.sc = None
        if spell_correction is True:
            if corrector is not None:
                self.sc = SpellCorrector(corpus=corrector)
            else:
                self.sc = SpellCorrector()

        self.ws = None
        if segmentation is True:
            if segmenter is not None:
                self.ws = Segmenter(corpus=segmenter)
            else:
                self.ws = Segmenter()

    def __spell_check(self, field_data):
        """
        Correct any spelling errors
        Args:
            field_data: text to correct
        Returns:
            field_data: correct text
        """

        def correct_word(word):
            if self.spell_correct_elong:
                # normalize to at most 2 repeating chars
                word = self.text_processor.regexes["normalize_elong"].sub(r'\1\1', word)

                normalized = self.sc.normalize_elongated(word)
                if normalized:
                    word = normalized

            return self.sc.correct_word(word, fast=True)

        return [correct_word(word) for word in field_data]

    def __word_segmenter(self, field_data) -> List[str]:
        """
        Split words together
        Args:
            field_data: Text to be processed
        Returns (List[str): Text with splitted words
        """
        word_seg_list = [self.ws.segment(word) for word in field_data]

        word_seg_list = itertools.chain.from_iterable([word.split() for word in word_seg_list])

        return list(word_seg_list)

    def process(self, field_data) -> List[str]:

        """
        Args:
            field_data: content to be processed
        Returns:
            field_data (List<str>): list of str or dict in case of named entity recognition
        """
        field_data = self.text_processor.pre_process_doc(field_data)
        if self.sc is not None:
            field_data = self.__spell_check(field_data)
        if self.ws is not None:
            field_data = self.__word_segmenter(field_data)
        return field_data

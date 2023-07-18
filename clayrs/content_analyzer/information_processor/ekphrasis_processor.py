import inspect
import itertools
from typing import List, Dict, Callable
import warnings

from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.segmenter import Segmenter

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.spellcorrect import SpellCorrector

from clayrs.content_analyzer.information_processor.information_processor_abstract import NLP
from clayrs.utils.automatic_methods import autorepr

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    social_tokenizer_ekphrasis = SocialTokenizer(lowercase=True).tokenize


class Ekphrasis(NLP):
    """
    Interface to the Ekphrasis library for natural language processing features

    Examples:

        * Normalize email and percentage tokens but omit email ones:
        >>> ek = Ekphrasis(omit=['email'], normalize=['email', 'percent'])
        >>> ek.process("this is an email: alias@mail.com and this is a percent 23%")
        ['this', 'is', 'an', 'email', ':', 'and', 'this', 'is', 'a', 'percent', '<percent>']

        * Unpack contractions on running text:
        >>> ek = Ekphrasis(unpack_contractions=True)
        >>> ek.process("I can't do this because I won't and I shouldn't")
        ['i', 'can', 'not', 'do', 'this', 'because', 'i', 'will', 'not', 'and', 'i', 'should', 'not']

        * Unpack hashtag using statistics from 'twitter' corpus:
        >>> ek = Ekphrasis(unpack_hashtags=True, segmenter='twitter')
        >>> ek.process("#next #gamedev #retrogaming #coolphoto no unpack")
        ['next', 'game', 'dev', 'retro', 'gaming', 'cool', 'photo', 'no', 'unpack']

        * Annotate words in CAPS and repeated tokens with single tag for CAPS words:
        >>> ek = Ekphrasis(annotate=['allcaps', 'repeated'], all_caps_tag='single')
        >>> ek.process("this is good !!! text and a SHOUTED one")
        ['this', 'is', 'good', '!', '<repeated>', 'text', 'and', 'a', 'shouted', '<allcaps>', 'one']

        * Perform segmentation using statistics from 'twitter' corpus:
        >>> ek = Ekphrasis(segmentation=True, segmenter='twitter')
        >>> ek.process("thewatercooler exponentialbackoff no segmentation")
        ['the', 'watercooler', 'exponential', 'back', 'off', 'no', 'segmentation']

        * Substitute words with custom tokens:
        >>> ek = Ekphrasis(dicts=[{':)': '<happy>', ':(': '<sad>'}])
        >>> ek.process("Hello :) how are you? :(")
        ['Hello', '<happy>', 'how', 'are', 'you', '?', '<sad>']

        * Perform spell correction on text and on elongated words by using statistics from default 'english' corpus:
        >>> Ekphrasis(spell_correction=True, spell_correct_elong=True)
        >>> ek.process("This is huuuuge. The korrect way of doing tihngs is not the followingt")
        ["this", 'is', 'huge', '.', 'the', 'correct', "way", "of", "doing", "things", "is", 'not', 'the',
        'following']

    Args:
        omit: Choose what tokens that you want to omit from the text.

            Possible values: ***['email', 'percent', 'money', 'phone', 'user','time', 'url', 'date', 'hashtag']***

            Important Notes:

                1 - the token in this list must be present in the `normalize`
                    list to have any effect!
                2 - put url at front, if you plan to use it.
                    Messes with the regexes!
                3 - if you use hashtag then unpack_hashtags will
                    automatically be set to False

        normalize: Choose what tokens that you want to normalize from the text.
            Possible values: ***['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'hashtag']***

            For example: myaddress@mysite.com -> `<email>`

            Important Notes:

                1 - put url at front, if you plan to use it.
                    Messes with the regexes!
                2 - if you use hashtag then unpack_hashtags will
                    automatically be set to False

        unpack_contractions: Replace *English* contractions in running text with their unshortened forms

            for example: can't -> can not, wouldn't -> would not, and so on...

        unpack_hashtags: split a hashtag to its constituent words.

            for example: #ilikedogs -> i like dogs

        annotate: add special tags to special tokens.

            Possible values: ['hashtag', 'allcaps', 'elongated', 'repeated']

            for example: myaddress@mysite.com -> myaddress@mysite.com <email>

        corrector: define the statistics of what corpus you would like to use [english, twitter].
            Be sure to set `spell_correction` to True if you want to perform
            spell correction on the running text

        tokenizer: callable function that accepts a string and returns a list of strings.
            If no tokenizer is provided then the text will be tokenized on whitespace

        segmenter: define the statistics of what corpus you would like to use [english, twitter].
            Be sure to set `segmentation` to True if you want to perform segmentation on the running text

        all_caps_tag: how to wrap the capitalized words
            Note: applicable only when `allcaps` is included in the `annotate` list
            Possible values ***[single, wrap, every]***:

                - single: add a tag after the last capitalized word
                    for example: "SHOUTED TEXT" -> "shouted text <allcaps>"
                - wrap: wrap all words with opening and closing tags
                    for example: "SHOUTED TEXT" -> "<allcaps> shouted text </allcaps>"
                - every: add a tag after each word
                    for example: "SHOUTED TEXT" -> "shouted <allcaps> text <allcaps>"

        spell_correction: If set to True, running text will be spell corrected using statistics of corpus set in
            `corrector` parameter

        segmentation: If set to True, running text will be segmented using statistics of corpus set in
            `corrector` parameter

            for example: exponentialbackoff -> exponential back off

        spell_correct_elong: choose if you want to perform spell correction after the normalization of elongated words.

            *significantly affects performance (speed)*

        spell_correction: choose if you want to perform spell correction to the text.

            *significantly affects performance (speed)*
    """

    def __init__(self, *,
                 omit: List = None,
                 normalize: List = None,
                 unpack_contractions: bool = False,
                 unpack_hashtags: bool = False,
                 annotate: List = None,
                 corrector: str = None,
                 tokenizer: Callable = social_tokenizer_ekphrasis,
                 segmenter: str = None,
                 all_caps_tag: str = None,
                 spell_correction: bool = False,
                 segmentation: bool = False,
                 dicts: List[Dict] = None,
                 spell_correct_elong: bool = False):

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

        self.segmentation = segmentation
        self.ws = None
        if segmentation is True:
            if segmenter is not None:
                self.ws = Segmenter(corpus=segmenter)
            else:
                self.ws = Segmenter()

        self._repr_string = autorepr(self, inspect.currentframe())

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

    def process(self, field_data: str) -> List[str]:
        """
        Args:
            field_data: Running text to be processed
        Returns:
            field_data: List of str representing running text preprocessed
        """
        field_data = self.text_processor.pre_process_doc(field_data)
        if self.sc is not None:
            field_data = self.__spell_check(field_data)
        if self.ws is not None:
            field_data = self.__word_segmenter(field_data)
        return field_data

    def __eq__(self, other):
        if isinstance(other, Ekphrasis):
            return self.text_processor.omit == other.text_processor.omit and \
                   self.text_processor.backoff == other.text_processor.backoff and \
                   self.text_processor.unpack_contractions == other.text_processor.unpack_contractions and \
                   self.text_processor.include_tags == other.text_processor.include_tags and \
                   self.text_processor.corrector_corpus == other.text_processor.corrector_corpus and \
                   self.text_processor.tokenizer == other.text_processor.tokenizer and \
                   self.text_processor.segmenter_corpus == other.text_processor.segmenter_corpus and \
                   self.text_processor.all_caps_tag == other.text_processor.all_caps_tag and \
                   self.text_processor.spell_correction == other.text_processor.spell_correction and \
                   self.segmentation == other.segmentation and \
                   self.text_processor.dicts == other.text_processor.dicts and \
                   self.spell_correct_elong == other.spell_correct_elong
        return False

    def __str__(self):
        return "Ekphrasis"

    def __repr__(self):
        return self._repr_string

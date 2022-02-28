import itertools
import string
from typing import List
import re

import nltk

from nltk import sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


class NLTK(NLP):
    _corpus_downloaded = False
    """
    Interface to the NLTK library for natural language processing features

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        stemming (bool): Whether you want to perform stemming
        lemmatization (bool): Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """

    def __init__(self, *,
                 strip_multiple_whitespaces: bool = True,
                 remove_punctuation: bool = False,
                 stopwords_removal: bool = False,
                 url_tagging: bool = False,
                 lemmatization: bool = False,
                 stemming: bool = False,
                 named_entity_recognition: bool = False,
                 lang: str = 'english'):

        self.__download_corpus()

        self.stopwords_removal = stopwords_removal
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.strip_multiple_whitespaces = strip_multiple_whitespaces
        self.url_tagging = url_tagging
        self.remove_punctuation = remove_punctuation
        self.named_entity_recognition = named_entity_recognition
        self.__full_lang_code = lang

    def __str__(self):
        return "NLTK"

    def __repr__(self):
        return "< NLTK: " \
               "stopwords_removal = " + \
               str(self.stopwords_removal) + ";" + \
               "stemming = " + \
               str(self.stemming) + ";" + \
               "lemmatization = " + \
               str(self.lemmatization) + ";" + \
               "named_entity_recognition = " + \
               str(self.named_entity_recognition) + ";" + \
               "strip_multiple_whitespaces = " + \
               str(self.strip_multiple_whitespaces) + ";" + \
               "url_tagging = " + \
               str(self.url_tagging) + \
               "punctuation_removal = " + \
               str(self.remove_punctuation) + \
               ">"

    def __download_corpus(self):
        if not self._corpus_downloaded:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt')
            try:
                nltk.data.find('averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            try:
                nltk.data.find('wordnet')
            except LookupError:
                nltk.download('wordnet')
            try:
                nltk.data.find('maxent_ne_chunker')
            except LookupError:
                nltk.download('maxent_ne_chunker')
            try:
                nltk.data.find('words')
            except LookupError:
                nltk.download('words')
            try:
                nltk.data.find('omw-1.4')
            except LookupError:
                nltk.download('omw-1.4')

        self._corpus_downloaded = True

    def __tokenization_operation(self, text) -> List[str]:
        """
        Splits the text in one-word tokens

        Args:
             text (str): Text to split in tokens

        Returns:
             List<str>: a list of words
        """
        # EXTREMELY useful tokenizer, it tokenizes by mantaining urls as a single token
        # as well as optional <url>, <hashtag>, etc.
        # It works for sentences so we first sentence tokenize
        sentences = sent_tokenize(text, self.__full_lang_code)
        sentences_tokenized = ToktokTokenizer().tokenize_sents(sentences)
        return list(itertools.chain.from_iterable(sentences_tokenized))

    def __stopwords_removal_operation(self, text) -> List[str]:
        """
        Execute stopwords removal on input text

        Args:
            text (List<str>):

        Returns:
            filtered_sentence (List<str>): list of words from the text, without the stopwords
        """
        stop_words = set(stopwords.words(self.__full_lang_code))
        filtered_sentence = []
        for word_token in text:
            if word_token.lower() not in stop_words:
                filtered_sentence.append(word_token)

        return filtered_sentence

    def __stemming_operation(self, text) -> List[str]:
        """
        Execute stemming on input text

        Args:
            text (List<str>):

        Returns:
            stemmed_text (List<str>): List of the fords from the text, reduced to their stem version
        """
        stemmer = SnowballStemmer(language=self.__full_lang_code)

        stemmed_text = [stemmer.stem(word) for word in text]

        return stemmed_text

    @staticmethod
    def __lemmatization_operation(text) -> List[str]:
        """
        Execute lemmatization on input text

        Args:
            text (List<str>):

        Returns:
            lemmatized_text (List<str>): List of the fords from the text, reduced to their lemmatized version
        """

        def get_wordnet_pos(word):
            """
            Map POS tag to first character lemmatize() accepts
            """
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        lemmatizer = WordNetLemmatizer()
        lemmatized_text = []
        for word in text:
            lemmatized_text.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        return lemmatized_text

    @staticmethod
    def __named_entity_recognition_operation(text) -> nltk.tree.Tree:
        """
        Execute NER on input text

        Args:
            text (List<str>): Text containing the entities

        Returns:
            namedEnt (nltk.tree.Tree): A tree containing the bonds between the entities
        """
        text = nltk.pos_tag(text)
        named_ent = nltk.ne_chunk(text)
        return named_ent

    @staticmethod
    def __strip_multiple_whitespaces_operation(text) -> str:
        """
        Remove multiple whitespaces on input text

        Args:
            text (str):

        Returns:
            str: input text, multiple whitespaces removed
        """
        return re.sub(' +', ' ', text)

    @staticmethod
    def __remove_punctuation(text) -> List[str]:
        """
        Punctuation removal in spacy
        Args:
            text (List[str]):
        Returns:
            string without punctuation
        """
        # remove all tokens that are not alphabetic
        cleaned_text = [word for word in text if word not in string.punctuation]
        return cleaned_text

    @staticmethod
    def __url_tagging_operation(text) -> List[str]:
        """
        Replaces urls with <URL> string on input text

        Args:
            text (List[str]):

        Returns:
            text (list<str>): input text, <URL> instead of full urls
        """
        tagged_token = []
        for token in text:
            if re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]| '
                        '[!*(), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        token):
                tagged_token.append("<URL>")
            else:
                tagged_token.append(token)

        return tagged_token

    def process(self, field_data) -> List[str]:
        field_data = check_not_tokenized(field_data)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        field_data = self.__tokenization_operation(field_data)
        if self.remove_punctuation:
            field_data = self.__remove_punctuation(field_data)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data)
        if self.stemming:
            field_data = self.__stemming_operation(field_data)
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data)
        return field_data

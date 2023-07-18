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

from clayrs.content_analyzer.information_processor.information_processor_abstract import NLP
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized


class NLTK(NLP):
    """
    Interface to the NLTK library for natural language processing features.
    
    Examples:

        * Strip multiple whitespaces from running text
        >>> nltk_obj = NLTK(strip_multiple_whitespaces=True)
        >>> nltk_obj.process('This   has  a lot  of   spaces')
        ['This', 'has', 'a', 'lot', 'of', 'spaces']

        * Remove punctuation from running text
        >>> nltk_obj = NLTK(remove_punctuation=True)
        >>> nltk_obj.process("Hello there. How are you? I'm fine, thanks.")
        ["Hello", "there", "How", "are", "you", "I", "m", "fine", "thanks"]

        * Remove stopwords using from running text
        >>> nltk_obj = NLTK(stopwords_removal=True)
        >>> nltk_obj.process("The striped bats are hanging on their feet for the best")
        ["striped", "bats", "hanging", "feet", "best"]

        * Replace URL with a normalized token `<URL>`
        >>> nltk_obj = NLTK(url_tagging=True)
        >>> nltk_obj.process("This is facebook http://facebook.com and github https://github.com")
        ['This', 'is', 'facebook', '<URL>', 'and', 'github', '<URL>']

        * Perform lemmatization on running text
        >>> nltk_obj = NLTK(lemmatization=True)
        >>> nltk_obj.process("The striped bats are hanging on their feet for best")
        ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"]

        * Perform stemming on running text
        >>> nltk_obj = NLTK(stemming=True)
        >>> nltk_obj.process("These unbelievable abnormous objects")
        ['these', 'unbeliev', 'abnorm', 'object']

        * Label each token in the running text with its POS tag
        >>> nltk_obj = NLTK(pos_tag=True)
        >>> nltk_obj.process("Facebook was fined by Hewlett Packard for spending 100€")
        ['Facebook_NNP', 'was_VBD', 'fined_VBN', 'by_IN', 'Hewlett_NNP', 'Packard_NNP', 'for_IN', 'spending_VBG',
        '100€_CD']
    
    Args:
        strip_multiple_whitespaces: If set to True, all multiple whitespaces will be reduced to only one white space
        remove_punctuation: If set to True, all punctuation from the running text will be removed
        stopwords_removal: If set to True, all stowpwords from the running text will be removed
        url_tagging: If set to True, all urls in the running text will be replaced with the `<URL>` token
        lemmatization: If set to True, each token in the running text will be brought to its lemma
        stemming: If set to True, each token in the running text will be brought to its stem
        pos_tag: If set to True, each token in the running text will be labeled with its POS tag in the form `token_TAG`
        lang: Language of the running text

    """
    _corpus_downloaded = False

    def __init__(self, *,
                 strip_multiple_whitespaces: bool = True,
                 remove_punctuation: bool = False,
                 stopwords_removal: bool = False,
                 url_tagging: bool = False,
                 lemmatization: bool = False,
                 stemming: bool = False,
                 pos_tag: bool = False,
                 lang: str = 'english'):

        if not NLTK._corpus_downloaded:
            self.__download_corpus()
            NLTK._corpus_downloaded = True

        self.stopwords_removal = stopwords_removal

        self.stemming = stemming
        self.stemmer = SnowballStemmer(language=lang)

        self.lemmatization = lemmatization
        self.lemmatizer = WordNetLemmatizer()

        self.strip_multiple_whitespaces = strip_multiple_whitespaces
        self.url_tagging = url_tagging
        self.remove_punctuation = remove_punctuation
        self.pos_tag = pos_tag
        self.__full_lang_code = lang

    def __download_corpus(self):
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
        stemmed_text = [self.stemmer.stem(word) for word in text]

        return stemmed_text

    def __lemmatization_operation(self, text) -> List[str]:
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

        lemmatized_text = []
        for word in text:
            lemmatized_text.append(self.lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        return lemmatized_text

    @staticmethod
    def __pos_operation(text) -> List[str]:
        """
        Execute POS on input text

        Args:
            text (List<str>): Text containing the entities

        Returns:
            namedEnt (nltk.tree.Tree): A tree containing the bonds between the entities
        """
        text_tuples = nltk.pos_tag(text)
        text_tagged = [f"{tagged[0]}_{tagged[1]}" for tagged in text_tuples]

        return text_tagged

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

    def process(self, field_data: str) -> List[str]:
        field_data = check_not_tokenized(field_data)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        field_data = self.__tokenization_operation(field_data)
        if self.remove_punctuation:
            field_data = self.__remove_punctuation(field_data)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data)
        if self.pos_tag:
            field_data = self.__pos_operation(field_data)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data)
        if self.stemming:
            field_data = self.__stemming_operation(field_data)
        return field_data

    def __eq__(self, other):
        if isinstance(other, NLTK):
            return self.strip_multiple_whitespaces == other.strip_multiple_whitespaces and \
                   self.remove_punctuation == other.remove_punctuation and \
                   self.stopwords_removal == other.stopwords_removal and \
                   self.url_tagging == other.url_tagging and \
                   self.lemmatization == other.lemmatization and \
                   self.stemming == other.stemming and \
                   self.pos_tag == other.pos_tag and \
                   self.__full_lang_code == other.__full_lang_code
        return False

    def __str__(self):
        return "NLTK"

    def __repr__(self):
        return f'NLTK(strip_multiple_whitespace={self.strip_multiple_whitespaces}, ' \
               f'remove_punctuation={self.remove_punctuation}, stopwords_removal={self.stopwords_removal}, ' \
               f'url_tagging={self.url_tagging}, lemmatization={self.lemmatization}, stemming={self.stemming}, ' \
               f'pos_tag={self.pos_tag}, ' \
               f'lang={self.__full_lang_code})'

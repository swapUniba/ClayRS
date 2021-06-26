from typing import List

import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer

from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


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


class NLTK(NLP):
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
    """
    Interface to the NLTK library for natural language processing features

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        stemming (bool): Whether you want to perform stemming
        lemmatization (bool): Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """
    def __init__(self, stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 lang='english'):

        if isinstance(stopwords_removal, str):
            stopwords_removal = stopwords_removal.lower() == 'true'

        if isinstance(stemming, str):
            stemming = stemming.lower() == 'true'

        if isinstance(lemmatization, str):
            lemmatization = lemmatization.lower() == 'true'

        if isinstance(strip_multiple_whitespaces, str):
            strip_multiple_whitespaces = strip_multiple_whitespaces.lower() == 'true'

        if isinstance(url_tagging, str):
            url_tagging = url_tagging.lower() == 'true'

        super().__init__(stopwords_removal,
                         stemming, lemmatization,
                         strip_multiple_whitespaces, url_tagging)

        self.__full_lang_code = lang

    def __str__(self):
        return "NLTK"

    def __repr__(self):
        return "< NLTK: " + "" \
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
               str(self.url_tagging) + " >"

    def set_lang(self, lang: str):
        self.lang = self.__full_lang_code

    def __tokenization_operation(self, text) -> List[str]:
        """
        Splits the text in one-word tokens

        Args:
             text (str): Text to split in tokens

        Returns:
             List<str>: a list of words
        """
        return [word for sent in nltk.sent_tokenize(text) for word in word_tokenize(sent)]

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

        stemmed_text = []
        for word in text:
            stemmed_text.append(stemmer.stem(word))

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

        lemmatizer = WordNetLemmatizer()
        lemmatized_text = []
        for word in text:
            lemmatized_text.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
        return lemmatized_text

    def __named_entity_recognition_operation(self, text) -> nltk.tree.Tree:
        """
        Execute NER on input text

        Args:
            text (List<str>): Text containing the entities

        Returns:
            namedEnt (nltk.tree.Tree): A tree containing the bonds between the entities
        """
        if type(text) == 'str':
            text = self.__tokenization_operation(text)
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
        import re
        return re.sub(' +', ' ', text)

    @staticmethod
    def __url_tagging_operation(text) -> List[str]:
        """
        Replaces urls with <URL> string on input text

        Args:
            text (str):

        Returns:
            text (list<str>): input text, <URL> instead of full urls
        """
        import re
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]| '
                          '[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                          text)
        for url in urls:
            text = text.replace(url, "<URL>")
        return text

    @staticmethod
    def __compact_tokens(text: List[str]) -> List[str]:
        """
        This method is useful because the tokenization operation separates the tokens that start with the '<'
        symbol. For example, the '<URL>' token is seen as three different tokens. This method brings together
        this kind of tokens, treating them as a unique one.

        Args:
            text (List<str>): List of tokens containing the tokens to compact

        Returns:
            text (List<str>): List of tokens in which the '<', 'URL', '>' tokens are compacted
                in an unique token
        """
        for i in range(0, len(text)):
            if i < len(text) and text[i] == '<':
                j = i + 1
                while text[j] != '>':
                    text[i] = text[i] + text[j]
                    del text[j]
                text[i] = text[i] + text[j]
                del text[j]
        return text

    def process(self, field_data) -> List[str]:
        field_data = check_not_tokenized(field_data)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data)
        field_data = self.__tokenization_operation(field_data)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data)
        if self.stemming:
            field_data = self.__stemming_operation(field_data)
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data)
        return self.__compact_tokens(field_data)

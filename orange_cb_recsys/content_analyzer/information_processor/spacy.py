from spacy.tests.conftest import tokenizer

from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP

from typing import List

import spacy

from spacy import tokens

from orange_cb_recsys.utils.check_tokenization import check_not_tokenized



class Spacy(NLP):
    """
    Interface to the Spacy library for natural language processing features

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        lemmatization (bool): Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """

    def __init__(self, lang='en_core_web_lg',
                 stopwords_removal: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 strip_multiple_whitespaces: bool = True,
                 url_tagging: bool = False,
                 ):


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
        return "Spacy"

    def __repr__(self):
        return "< Spacy: " + "" \
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

    def __tokenization_operation(self, text, nlp) -> List[str]:
        """
        Splits the text in one-word tokens

        Args:
             text (str): Text to split in tokens

        Returns:
             List<str>: a list of words
        """
        doc = nlp(text)
        return [token.text for token in doc]

    def __stopwords_removal_operation(self, text, nlp) -> List[str]:
        """
        Execute stopwords removal on input text with spacy

        Args:
            text (List<str>):

        Returns:
            filtered_sentence (List<str>): list of words from the text, without the stopwords
        """
        stringText = self.__listTostring(text)
        filtered_sentence = []
        for token in nlp(stringText):
            if not (token.is_stop):
                filtered_sentence.append(token.text)
        return filtered_sentence


    def __stemming_operation(self, text, nlp) -> List[str]:
        pass

    def __lemmatization_operation(self, text, nlp) -> List[str]:
        """
        Execute lemmatization on input text with spacy

        Args:
            text (List<str>):

        Returns:
            lemmatized_text (List<str>): List of the fords from the text, reduced to their lemmatized version
        """

        stringText = self.__listTostring(text)
        lemmatized_text = []
        for word in nlp(stringText):
            lemmatized_text.append(word.lemma_)
        return lemmatized_text


    def named_entity_recognition_operation(self, text):
        nlp=spacy.load(self.__full_lang_code)
        for token in nlp(text):
            print(token.tag_)
            #continue...

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


    def __url_tagging_operation(self,text, nlp) -> List[str]:
        """
                Replaces urls with <URL> string on input text with spacy

                Args:
                    text (str):

                Returns:
                    text (list<str>): input text, <URL> instead of full urls
                """
        textURL=[]
        stringText = self.__listTostring(text)
        for i, token in enumerate(nlp(stringText)):
            if token.like_url:
                token.tag_ = "<URL>"
                textURL.append(token.tag_)
            else:
                textURL.append(token)
        return textURL

    @staticmethod
    def __listTostring(text: List[str])->str:
        """
        Covert list of str in str
        Args:
            text: list of str

        Returns: str sentence

        """
        stringText = ' '.join([str(elem) for elem in text])
        return stringText

    @staticmethod
    def __tokenToString(tokenField, nlp) -> str:
        """
        After preprocessing with spacy the output was given as a list of str

        Args:
            tokenField: List of tokens
        Returns:
            list of string
        """
        stringList=[]
        for token in tokenField:
            stringList.append(str(token))
        return stringList

    def process(self, field_data) -> List[str]:
        field_data = check_not_tokenized(field_data)
        nlp=spacy.load(self.__full_lang_code)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        field_data=self.__tokenization_operation(field_data, nlp)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data, nlp)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data,nlp)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data, nlp)
        if self.stemming:
            field_data = self.__stemming_operation(field_data,nlp)
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data)
        field_data=self.__tokenToString(field_data, nlp)

        return field_data

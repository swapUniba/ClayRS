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
                 named_entity_recognition: bool = False,
                 remove_punctuation: bool = False
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

        if isinstance(named_entity_recognition,str):
            named_entity_recognition = named_entity_recognition() == 'true'

        if isinstance(remove_punctuation, str):
            remove_punctuation = remove_punctuation.lower() == 'true'


        super().__init__(stopwords_removal,
                         stemming, lemmatization,
                         strip_multiple_whitespaces, url_tagging,
                         #named_entity_recognition,
                         remove_punctuation)


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

    @staticmethod
    def __named_entity_recognition_operation(self, text, nlp):
        """
                Execute NER on input text with spacy
                Args:
                    text List[str]: Text containing the entities
                Returns:
                    word_entity: Dict of entity
        """
        word_entity = {}
        string_text = self.__list_to_string(text)
        for value, token in enumerate(nlp(string_text)):
            word_entity[value] = [token, token.tag_]
        return word_entity



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


    @staticmethod
    def __list_to_string(text: List[str]) -> str:
        """
        Covert list of str in str
        Args:
            text: list of str
        Returns: str sentence
        """
        string_text = ' '.join([str(elem) for elem in text])
        return string_text

    @staticmethod
    def __token_to_string(token_field) -> List[str]:
        """
        After preprocessing with spacy the output was given as a list of str
        Args:
            token_field: List of tokens
        Returns:
            list of string
        """
        string_list = []
        for token in token_field:
            string_list.append(str(token))
        return string_list

    def __check_if_download(self):
        """
        check if the model already exists to load it.
        If it doesn't exist, download it
        Returns: nlp
        """
        if self.__full_lang_code not in spacy.cli.info()['pipelines']:
            spacy.cli.download(self.__full_lang_code)
        nlp = spacy.load(self.__full_lang_code)
        return nlp


    def __remove_punctuation(self, text, nlp) -> str:
        """
        Punctuation removal in spacy
        Args:
            text (str):
        Returns:
            string without punctuation
        """
        text = self.__list_to_string(text)
        cleaned_text = []
        for token in nlp(text):
            if not (token.is_punct):
                cleaned_text.append(token.text)
        cleaned_text = self.__listTostring(cleaned_text)
        return cleaned_text



    def process(self, field_data) -> List[str]:
        """
        Args:
            field_data: content to be processed
        Returns:
            field_data: list of str or dict in case of named entity recognition
        """
        nlp = self.__check_if_download()
        field_data = check_not_tokenized(field_data)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        field_data = self.__tokenization_operation(field_data, nlp)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data, nlp)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data, nlp)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data, nlp)
        if self.stemming:
            field_data = self.__stemming_operation(field_data, nlp)
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data, nlp)
        if self.__remove_punctuation:
            field_data = self.__remove_punctuation(field_data, nlp)
            return field_data
        else:
            return self.__token_to_string(field_data)
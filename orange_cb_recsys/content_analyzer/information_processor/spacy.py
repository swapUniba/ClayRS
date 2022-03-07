from spacy.tokens import Token

from orange_cb_recsys.content_analyzer.information_processor.information_processor import NLP

from typing import List, Dict

from orange_cb_recsys.utils.check_tokenization import check_not_tokenized

import spacy


class Spacy(NLP):
    """
    Interface to the Spacy library for natural language processing features

    Args:
        stopwords_removal (bool): Whether you want to remove stop words
        lemmatization (bool): Whether you want to perform lemmatization
        strip_multiple_whitespaces (bool): Whether you want to remove multiple whitespaces
        url_tagging (bool): Whether you want to tag the urls in the text and to replace with "<URL>"
    """

    def __init__(self, model: str = 'en_core_web_sm', *,
                 strip_multiple_whitespaces: bool = True,
                 remove_punctuation: bool = False,
                 stopwords_removal: bool = False,
                 new_stopwords: List[str] = None,
                 not_stopwords: List[str] = None,
                 lemmatization: bool = False,
                 url_tagging: bool = False,
                 named_entity_recognition: bool = False,
                 ):
        self.stopwords_removal = stopwords_removal
        self.lemmatization = lemmatization
        self.strip_multiple_whitespaces = strip_multiple_whitespaces
        self.url_tagging = url_tagging
        self.remove_punctuation = remove_punctuation
        self.named_entity_recognition = named_entity_recognition

        # download the model if not present. In any case load it
        if model not in spacy.cli.info()['pipelines']:
            spacy.cli.download(model)
        self._nlp = spacy.load(model)

        # Adding custom rule of preserving <URL> token and in general token
        # wrapped by <...>
        prefixes = list(self._nlp.Defaults.prefixes)
        prefixes.remove('<')
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self._nlp.tokenizer.prefix_search = prefix_regex.search

        suffixes = list(self._nlp.Defaults.suffixes)
        suffixes.remove('>')
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        self._nlp.tokenizer.suffix_search = suffix_regex.search

        if not_stopwords is not None:
            for stopword in not_stopwords:
                self._nlp.vocab[stopword].is_stop = False

        if new_stopwords is not None:
            for stopword in new_stopwords:
                self._nlp.vocab[stopword].is_stop = True

    def __str__(self):
        return "Spacy"

    def __repr__(self):
        return f'NLTK(model={str(self._nlp)}, strip multiple whitespace={self.strip_multiple_whitespaces}, ' \
               f'stopwords removal={self.stopwords_removal},' \
               f'lemmatization={self.lemmatization},' \
               f' url tagging={self.url_tagging}, remove punctuation={self.remove_punctuation},' \
               f' named entity recognition={self.named_entity_recognition})'

    def __tokenization_operation(self, text) -> List[Token]:
        """
        Splits the text in one-word tokens

        Args:
             text (str): Text to split in tokens

        Returns:
             List<str>: a list of words
        """
        pipe_to_disable = ['tagger', 'parser', 'textcat']

        if not self.lemmatization:
            pipe_to_disable.append('lemmatizer')
        if not self.named_entity_recognition:
            pipe_to_disable.append('ner')

        return list(self._nlp(text))

    def __stopwords_removal_operation(self, text) -> List[Token]:
        """
        Execute stopwords removal on input text with spacy

        Args:
            text (List[Token]):

        Returns:
            filtered_sentence (List<str>): list of words from the text, without the stopwords
        """
        filtered_sentence = [token for token in text if not token.is_stop]

        return filtered_sentence

    def __lemmatization_operation(self, text) -> List[Token]:
        """
        Execute lemmatization on input text with spacy

        Args:
            text (List[Token]):

        Returns:
            lemmatized_text (List<str>): List of the fords from the text, reduced to their lemmatized version
        """
        lemmas_to_tokenize = ' '.join([word.lemma_ for word in text])

        return self.__tokenization_operation(lemmas_to_tokenize)

    def __named_entity_recognition_operation(self, text) -> Dict[str, str]:
        """
        Execute NER on input text with spacy

        Args:
            text List[str]: Text containing the entities

        Returns:
            word_entity: Dict of entity
        """
        tokens_with_entities = self._nlp(' '.join([str(word) for word in text]))
        word_entity = {word.text: word.label_ for word in tokens_with_entities.ents}

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

    def __url_tagging_operation(self, text) -> List[Token]:
        """
        Replaces urls with <URL> string on input text with spacy

        Args:
            text (list[Token]):

        Returns:
            text (list<str>): input text, <URL> instead of full urls
        """

        text_w_url_to_tokenize = ' '.join(["<URL>" if token.like_url else str(token) for token in text])

        return self.__tokenization_operation(text_w_url_to_tokenize)

    def __remove_punctuation(self, text) -> List[Token]:
        """
        Punctuation removal in spacy
        Args:
            text (list[Token]):
        Returns:
            string without punctuation
        """
        cleaned_text = [token for token in text if not token.is_punct]

        return cleaned_text

    @staticmethod
    def __token_to_string(token_field) -> List[str]:
        """
        After preprocessing with spacy the output was given as a list of str

        Args:
            token_field: List of tokens
        Returns:
            list of string
        """
        string_list = [str(token) for token in token_field]

        return string_list

    def process(self, field_data) -> List[str]:
        """
        Args:
            field_data: content to be processed

        Returns:
            field_data: list of str or dict in case of named entity recognition

        """
        field_data = check_not_tokenized(field_data)
        if self.strip_multiple_whitespaces:
            field_data = self.__strip_multiple_whitespaces_operation(field_data)
        field_data = self.__tokenization_operation(field_data)
        if self.remove_punctuation:
            field_data = self.__remove_punctuation(field_data)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data)
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data)
            return field_data
        else:
            return self.__token_to_string(field_data)

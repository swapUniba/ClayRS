from typing import List
import warnings

# spacy has a bug which prints a useless warning if pytorch cuda is installed but no gpu is detected
with warnings.catch_warnings():

    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    import spacy
    from spacy.tokens import Token

from clayrs.content_analyzer.information_processor.information_processor_abstract import NLP
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized


class Spacy(NLP):
    """
    Interface to the Spacy library for natural language processing features

    Examples:

        * Strip multiple whitespaces from running text
        >>> spacy_obj = Spacy(strip_multiple_whitespaces=True)
        >>> spacy_obj.process('This   has  a lot  of   spaces')
        ['This', 'has', 'a', 'lot', 'of', 'spaces']

        * Remove punctuation from running text
        >>> spacy_obj = Spacy(remove_punctuation=True)
        >>> spacy_obj.process("Hello there. How are you? I'm fine, thanks.")
        ["Hello", "there", "How", "are", "you", "I", "'m", "fine", "thanks"]

        * Remove stopwords using default stopwords corpus of spacy from running text
        >>> spacy_obj = Spacy(stopwords_removal=True)
        >>> spacy_obj.process("The striped bats are hanging on their feet for the best")
        ["striped", "bats", "hanging", "feet", "best"]

        * Remove stopwords using default stopwords corpus of spacy + `new_stopwords` list from running text
        >>> spacy_obj = Spacy(stopwords_removal=True, new_stopwords=['bats', 'best'])
        >>> spacy_obj.process("The striped bats are hanging on their feet for the best")
        ["striped", "hanging", "feet"]

        * Remove stopwords using default stopwords corpus of spacy - `not_stopwords` list from running text
        >>> spacy_obj = Spacy(stopwords_removal=True, not_stopwords=['The', 'the', 'on'])
        >>> spacy_obj.process("The striped bats are hanging on their feet for the best")
        ["The", "striped", "bats", "hanging", "on", "feet", "the", "best"]

        * Replace URL with a normalized token `<URL>`
        >>> spacy_obj = Spacy(url_tagging=True)
        >>> spacy_obj.process("This is facebook http://facebook.com and github https://github.com")
        ['This', 'is', 'facebook', '<URL>', 'and', 'github', '<URL>']

        * Perform lemmatization on running text
        >>> spacy_obj = Spacy(lemmatization=True)
        >>> spacy_obj.process("The striped bats are hanging on their feet for best")
        ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"]

        * Perform NER on running text (NEs will be tagged with BIO tagging)
        >>> spacy_obj = Spacy(named_entity_recognition=True)
        >>> spacy_obj.process("Facebook was fined by Hewlett Packard for spending 100€")
        ["Facebook", "was", "fined", "by", "<Hewlett_ORG_B>", "<Packard_ORG_I>", "for", "spending",
        "<100_MONEY_B>", "<€_MONEY_I>"]

    Args:
        model: Spacy model that will be used to perform nlp operations. It will be downloaded if not present locally
        strip_multiple_whitespaces: If set to True, all multiple whitespaces will be reduced to only one white space
        remove_punctuation: If set to True, all punctuation from the running text will be removed
        stopwords_removal: If set to True, all stowpwords from the running text will be removed
        new_stopwords: List which contains custom defined stopwords that will be removed if `stopwords_removal=True`
        not_stopwords: List which contains custom defined stopwords that will not be considered as such, therefore won't
            be removed if `stopwords_removal=True`
        url_tagging: If set to True, all urls in the running text will be replaced with the `<URL>` token
        lemmatization: If set to True, each token in the running text will be brought to its lemma
        named_entity_recognition: If set to True, named entities recognized will be labeled in the form `<token_B_TAG>`
            or `<token_I_TAG>`, according to BIO tagging strategy
    """

    def __init__(self, model: str = 'en_core_web_sm', *,
                 strip_multiple_whitespaces: bool = True,
                 remove_punctuation: bool = False,
                 stopwords_removal: bool = False,
                 new_stopwords: List[str] = None,
                 not_stopwords: List[str] = None,
                 lemmatization: bool = False,
                 url_tagging: bool = False,
                 named_entity_recognition: bool = False):

        self.model = model
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

        # Adding custom rule of preserving '<URL>' token and in general token
        # wrapped by '<...>'
        prefixes = list(self._nlp.Defaults.prefixes)
        prefixes.remove('<')
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        self._nlp.tokenizer.prefix_search = prefix_regex.search

        suffixes = list(self._nlp.Defaults.suffixes)
        suffixes.remove('>')
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        self._nlp.tokenizer.suffix_search = suffix_regex.search

        self.not_stopwords_list = not_stopwords
        if not_stopwords is not None:
            for stopword in not_stopwords:
                self._nlp.vocab[stopword].is_stop = False

        self.new_stopwords_list = new_stopwords
        if new_stopwords is not None:
            for stopword in new_stopwords:
                self._nlp.vocab[stopword].is_stop = True

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

    def __named_entity_recognition_operation(self, text) -> List[Token]:
        """
        Execute NER on input text with spacy

        Args:
            text List[str]: Text containing the entities

        Returns:
            word_entity: Dict of entity
        """
        labeled_entities = ' '.join([f"<{token.text}_{token.ent_type_}_{token.ent_iob_}>" if token.ent_type != 0
                                     else f"{token.text}" for token in text])

        return self.__tokenization_operation(labeled_entities)

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
        string_list = [token.text for token in token_field]

        return string_list

    def process(self, field_data: str) -> List[str]:
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
        if self.named_entity_recognition:
            field_data = self.__named_entity_recognition_operation(field_data)
        if self.remove_punctuation:
            field_data = self.__remove_punctuation(field_data)
        if self.stopwords_removal:
            field_data = self.__stopwords_removal_operation(field_data)
        if self.lemmatization:
            field_data = self.__lemmatization_operation(field_data)
        if self.url_tagging:
            field_data = self.__url_tagging_operation(field_data)

        return self.__token_to_string(field_data)

    def __eq__(self, other):
        if isinstance(other, Spacy):
            return self.model == other.model and \
                   self.strip_multiple_whitespaces == other.strip_multiple_whitespaces and \
                   self.remove_punctuation == other.remove_punctuation and \
                   self.stopwords_removal == other.stopwords_removal and \
                   self.new_stopwords_list == other.new_stopwords_list and \
                   self.not_stopwords_list == other.not_stopwords_list and \
                   self.lemmatization == other.lemmatization and \
                   self.url_tagging == other.url_tagging and \
                   self.named_entity_recognition == other.named_entity_recognition
        return False

    def __str__(self):
        return "Spacy"

    def __repr__(self):
        return f'Spacy(model={self.model}, strip_multiple_whitespace={self.strip_multiple_whitespaces}, ' \
               f'remove_punctuation={self.remove_punctuation}, stopwords_removal={self.stopwords_removal}, ' \
               f'new_stopwords={self.new_stopwords_list}, not_stopwords={self.not_stopwords_list}, ' \
               f'lemmatization={self.lemmatization}, url_tagging={self.url_tagging}, ' \
               f'named_entity_recognition={self.named_entity_recognition})'

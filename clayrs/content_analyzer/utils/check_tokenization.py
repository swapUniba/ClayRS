from typing import Union, List
from nltk import data, download, sent_tokenize, word_tokenize

# nltk corpus
corpus_downloaded = False


def check_tokenized(text):
    """
    Tokenizes a text
    """
    if type(text) is str:
        global corpus_downloaded

        if not corpus_downloaded:
            try:
                data.find('punkt')
            except LookupError:
                download('punkt')

            corpus_downloaded = True

        text = word_tokenize(text)

    return text


def check_not_tokenized(text):
    """
    Untokenizes a tokenized text
    """
    if type(text) is list:
        text = ' '.join(text)

    return text


def tokenize_in_sentences(text: Union[List[str], str]):
    """
    Tokenizes a text into sentences
    """
    global corpus_downloaded

    if not corpus_downloaded:
        try:
            data.find('punkt')
        except LookupError:
            download('punkt')

        corpus_downloaded = True

    return sent_tokenize(check_not_tokenized(text))

from typing import Union, List
from nltk import RegexpTokenizer, data, download, sent_tokenize

try:
    data.find('punkt')
except LookupError:
    download('punkt')


def check_tokenized(text):
    """
    Tokenizes a text
    """
    if type(text) is str:
        tokenizer = RegexpTokenizer('[\w<>$€]+')
        text = tokenizer.tokenize(text)

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

    return sent_tokenize(check_not_tokenized(text))

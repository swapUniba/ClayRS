from nltk import RegexpTokenizer


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

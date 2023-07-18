from unittest import TestCase
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized, check_tokenized, tokenize_in_sentences


class Test(TestCase):
    def test_check_tokenized(self):
        str_ = 'abcd efg'
        list_ = ['abcd', 'efg']
        check_tokenized(str_)
        check_tokenized(list_)
        check_not_tokenized(str_)
        check_not_tokenized(list_)

    def test_tokenize_sentence(self):

        phrases = "Ciao, questa è una prova. Anche questa. And this is the third"
        result = tokenize_in_sentences(phrases)

        self.assertTrue(len(result) == 3)
from unittest import TestCase
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized, check_tokenized


class Test(TestCase):
    def test_check_tokenized(self):
        str_ = 'abcd efg'
        list_ = ['abcd', 'efg']
        check_tokenized(str_)
        check_tokenized(list_)
        check_not_tokenized(str_)
        check_not_tokenized(list_)
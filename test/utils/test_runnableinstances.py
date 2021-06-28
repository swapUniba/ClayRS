import unittest
from unittest import TestCase
# from orange_cb_recsys.utils.runnable_instances import *

@unittest.skip("Script not yet implemented")
class Test(TestCase):

    def test_runnable_instances(self):
        show()

        get()

        add('test', 'test_test')

        remove('test')

        show()

        add('test2', 'test_cat', 'preprocessor')

        try:
            add('test2', 'test_cat', 'test_fail')
        except ValueError:
            pass

        show(True)

        print(get_cat('preprocessor'))

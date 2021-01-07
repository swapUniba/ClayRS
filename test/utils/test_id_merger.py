from unittest import TestCase

from orange_cb_recsys.utils.id_merger import id_values_merger


class Test(TestCase):
    def test_id_merger(self):
        self.assertEqual(id_values_merger('aaa'), 'aaa', "Must return a string value")
        self.assertEqual(id_values_merger(['aaa', 'bbb']), 'aaa_bbb', "Must return a string value like this aaa_bbb")
        self.assertEqual(id_values_merger(123), '123', "Must return a string value")
        self.assertEqual(id_values_merger([123, 124]), '123_124', "Must return a string value like this 123_124")
        self.assertEqual(id_values_merger([123, "aaa"]), '123_aaa', "Must return a string value like 123_aaa")
        with self.assertRaises(TypeError):
            id_values_merger({1: 1, 2: 2})

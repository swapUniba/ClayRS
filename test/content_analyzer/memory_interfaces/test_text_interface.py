from unittest import TestCase

from orange_cb_recsys.content_analyzer.memory_interfaces import KeywordIndex, SearchIndex


class TestTextInterface(TestCase):
    def test_keyword_serialize(self):
        index = KeywordIndex("/keyword")
        try:
            index.init_writing()
            index.new_content()
            index.new_field("content_id", "0")
            index.new_field("test1", ["this", "is", "a", "test"])
            index.new_field("test2", ["this", "is", "a", "test", "for", "the", "Text", "Interface"])
            index.serialize_content()
            index.stop_writing()
            self.assertEqual(index.get_field("test1", "0"), ["this", "is", "a", "test"])
            self.assertEqual(index.get_tf_idf("test1", "0")["this"], 0.0)
        finally:
            index.delete()

    def test_search_serialize(self):
        index = SearchIndex("search")
        try:
            index.init_writing()
            index.new_content()
            index.new_field("content_id", "0")
            index.new_field("test1", "This is A test")
            index.new_field("test2", "this is a test for the Text Interface")
            index.serialize_content()
            index.stop_writing()
            self.assertEqual(index.get_field("test1", "0"), "This is A test")
        finally:
            index.delete()

    def test_init_writing(self):
        index1 = SearchIndex("init_writing")
        index2 = SearchIndex("init_writing")

        try:
            index1.init_writing()
            index1.new_content()
            index1.new_field("content_id", "0")
            index1.new_field("init_writing", "test")
            index1.serialize_content()
            index1.stop_writing()

            index2.init_writing()
            self.assertEqual(index2.get_field("init_writing", "0"), "test")
        finally:
            index1.delete()
            index2.delete()



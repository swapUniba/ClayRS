from unittest import TestCase

from orange_cb_recsys.content_analyzer.memory_interfaces import KeywordIndex, SearchIndex


class TestIndexInterface(TestCase):
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
        index1 = SearchIndex("./init_writing")
        index2 = SearchIndex("./init_writing")
        index3 = SearchIndex("./init_writing")

        try:
            index1.init_writing()
            index1.new_content()
            index1.new_field("content_id", "0")
            index1.new_field("init_writing", "test1")
            index1.serialize_content()
            index1.stop_writing()

            # test to check that init_writing with False argument doesn't replace the old index but opens it
            index2.init_writing(False)
            index2.new_content()
            index2.new_field("content_id", "1")
            index2.new_field("init_writing", "test2")
            index2.serialize_content()
            index2.stop_writing()
            self.assertEqual(index2.get_field("init_writing", "0"), "test1")
            self.assertEqual(index2.get_field("init_writing", "1"), "test2")

            # test to check that init_writing with True argument replaces the old index
            index3.init_writing(True)
            index3.new_content()
            index3.new_field("content_id", "0")
            index3.new_field("init_writing", "test3")
            index3.serialize_content()
            index3.stop_writing()
            self.assertEqual(index3.get_field("init_writing", "0"), "test3")
            with self.assertRaises(IndexError):
                index3.get_field("init_writing", "1")
        finally:
            index1.delete()
            index2.delete()
            index3.delete()

    def test_get_field(self):
        index = KeywordIndex("/keyword")

        try:
            index.init_writing()
            index.new_content()
            index.new_field("content_id", "0")
            index.new_field("test1", ["this", "is", "a", "test"])
            index.new_field("test2", ["this", "is", "a", "test", "for", "the", "Text", "Interface"])
            index.serialize_content()
            index.stop_writing()

            # test for using get_field with content's id
            result = index.get_field("test1", "0")
            self.assertIsInstance(result, list)
            self.assertEqual(result, ["this", "is", "a", "test"])

            # test for using get_field with content's position in the index
            result = index.get_field("test1", 0)
            self.assertIsInstance(result, list)
            self.assertEqual(result, ["this", "is", "a", "test"])
        finally:
            index.delete()

    def test_get_tfidf(self):
        index = KeywordIndex("/keyword")
        try:
            index.init_writing()
            index.new_content()
            index.new_field("content_id", "0")
            index.new_field("test1", ["this", "is", "a", "test"])
            index.new_field("test2", ["this", "is", "a", "test", "for", "the", "Text", "Interface"])
            index.serialize_content()
            index.stop_writing()

            # test for using get_tf_idf with content's id
            result = index.get_tf_idf("test1", "0")
            self.assertIsInstance(result, dict)
            self.assertEqual(result["this"], 0.0)

            # test for using get_td_idf with content's position in the index
            result = index.get_tf_idf("test1", 0)
            self.assertIsInstance(result, dict)
            self.assertEqual(result["this"], 0.0)
        finally:
            index.delete()

    def test_query(self):
        index = SearchIndex("testing_query")
        try:
            index.init_writing()
            index.new_content()
            index.new_field("content_id", "0")
            index.new_field("test1", "this is a test for the query on the index")
            index.new_field("test2", "this is the second field")
            index.serialize_content()
            index.new_content()
            index.new_field("content_id", "1")
            index.new_field("test1", "field")
            index.serialize_content()
            index.new_content()
            index.new_field("content_id", "2")
            index.new_field("test1", "query on the index")
            index.serialize_content()
            index.stop_writing()

            # test for querying the index
            result = index.query("test1:(query on the index)", 2, ["2"], ["0", "1"], True)
            self.assertEqual(len(result), 1)
            self.assertEqual(result["0"]["item"]["test1"], "this is a test for the query on the index")
        finally:
            index.delete()


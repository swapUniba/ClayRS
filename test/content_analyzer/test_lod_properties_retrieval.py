from unittest import TestCase

from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset


class TestDBPediaMappingTechnique(TestCase):
    def test_get_properties(self):

        raw_content = {"Title": "Jumanji", "Year": "1995", "Rated": "PG",
                       "Released": "15 Dec 1995", "Budget_source": "6.5E7",
                       "cinematography": "", "only_local": "", "wiki_id": "3700174"}

        mapp = DBPediaMappingTechnique('Film', 'EN', 'Title')

        # Get properties THAT HAVE VALUE in DBPEDIA ONLY
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertNotIn("only_local", prop.value)

        # Get all properties in DBPEDIA + Get all properties IN LOCAL SOURCE
        mapp.mode = 'all'
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value["completion_date"], "")
        self.assertEqual(prop.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertEqual(prop.value["only_local"], "")
        self.assertEqual(prop.value["Title"], "Jumanji")

        # Get all properties in DBPEDIA ONLY
        mapp.mode = 'all_retrieved'
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value["completion_date"], "")
        self.assertEqual(prop.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertNotIn("only_local", prop.value)

        # Get all properties in LOCAL with value from DBPEDIA
        mapp.mode = 'original_retrieved'
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertEqual(prop.value["only_local"], '')

        # additional_filter Query
        mapp = DBPediaMappingTechnique('Film', 'EN', 'Title', additional_filters={'budget': 'Budget_source', 'wikiPageID': 'wiki_id'})
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')


class TestPropertiesFromDataset(TestCase):
    def test_get_properties(self):
        raw_content = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995",
                       "Runtime": ""}

        # mode is 'only_retrieved_evaluated' so all item with a blank value
        # will be discarded
        expected_no_blank = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995"}
        mapp = PropertiesFromDataset()
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value, expected_no_blank)

        # mode is 'all' so all item are fine, also those with
        # a blank value
        expected_w_blank = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995",
                            "Runtime": ""}
        mapp.mode = 'all'
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value, expected_w_blank)

        # field_name_list with 'title' is passed, and mode is 'only_retrieved_evaluated',
        # so only the field 'title' will be retrieved, if it has a value
        expected_title_only = {'Title': 'Jumanji'}
        mapp = PropertiesFromDataset(field_name_list=['Title'])
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value, expected_title_only)

        # field_name_list with a nonexistent field is passed,
        # since the mode is 'only_retrieved_evaluated' nothing will be retrieved
        expected_empty = {}
        mapp = PropertiesFromDataset(field_name_list=['pppp'])
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value, expected_empty)

        # field_name_list with a nonexistent field is passed,
        # since the mode is 'all' everything is retrieved, also the nonexistent field.
        # It will have a blank value
        expected_nonexistant = {'pppp': ''}
        mapp.mode = "all"
        prop = mapp.get_properties(raw_content)
        self.assertEqual(prop.value, expected_nonexistant)
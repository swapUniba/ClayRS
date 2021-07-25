import os
from unittest import TestCase

from orange_cb_recsys.content_analyzer import JSONFile
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset
from orange_cb_recsys.utils.const import datasets_path

source_path = os.path.join(datasets_path, 'test_dbpedia', 'movies_info_reduced.json')


class TestDBPediaMappingTechnique(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_source = JSONFile(source_path)

    def test_get_properties_as_labels(self):

        mapp = DBPediaMappingTechnique('dbo:Film', 'Title')

        # Get properties THAT HAVE VALUE in DBPEDIA ONLY
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertIsInstance(prop1.value["starring"], list)
        self.assertNotIn("only_local", prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value["film director"], "http://dbpedia.org/resource/Christopher_Nolan")

        # Get all properties in DBPEDIA + Get all properties IN LOCAL SOURCE
        # Local properties will be overwritten by dbpedia values if there's a conflict
        mapp.mode = 'all'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value["completion date"], "")
        self.assertEqual(prop1.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertIsInstance(prop1.value["starring"], list)
        self.assertEqual(prop1.value["only_local"], "")
        self.assertEqual(prop1.value["Title"], "Jumanji")

        prop2 = results[1]
        self.assertEqual(prop2.value["film director"], "http://dbpedia.org/resource/Christopher_Nolan")
        self.assertEqual(prop2.value["only_local"], "")

        # Get all properties in DBPEDIA ONLY
        mapp.mode = 'all_retrieved'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value["completion date"], "")
        self.assertEqual(prop1.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertIsInstance(prop1.value["starring"], list)
        self.assertNotIn("only_local", prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value["completion date"], "")
        self.assertEqual(prop2.value["film director"], "http://dbpedia.org/resource/Christopher_Nolan")
        self.assertIsInstance(prop2.value["starring"], list)
        self.assertNotIn("only_local", prop2.value)

        # Get all properties in LOCAL with value from DBPEDIA
        mapp.mode = 'original_retrieved'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertEqual(prop1.value["only_local"], '')
        self.assertNotIn('starring', prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value["Title"], '')
        self.assertEqual(prop2.value["Budget_source"], '')
        self.assertEqual(prop2.value["only_local"], '')

        # # additional_filter Query
        # mapp = DBPediaMappingTechnique('dbo:Film', 'Title', additional_filters={'dbo:budget': 'Budget_source',
        #                                                                           'dbo:wikiPageID': 'wiki_id'})
        # results = mapp.get_properties(self.raw_source)
        #
        # prop1 = results[0]
        # self.assertEqual(prop1.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        #
        # prop2 = results[1]
        # self.assertEqual(prop2.value["film director"], "http://dbpedia.org/resource/Christopher_Nolan")

    def test_get_properties_as_uri(self):

        mapp = DBPediaMappingTechnique('dbo:Film', 'Title', prop_as_uri=True)

        uri_prop_cinematography = "http://dbpedia.org/ontology/cinematography"
        uri_prop_starring = "http://dbpedia.org/ontology/starring"
        uri_prop_completionDate = "http://dbpedia.org/ontology/completionDate"
        uri_prop_director = "http://dbpedia.org/ontology/director"

        # Get properties THAT HAVE VALUE in DBPEDIA ONLY
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value[uri_prop_cinematography], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertIsInstance(prop1.value[uri_prop_starring], list)
        self.assertNotIn("only_local", prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value[uri_prop_director], "http://dbpedia.org/resource/Christopher_Nolan")

        # Get all properties in DBPEDIA + Get all properties IN LOCAL SOURCE
        # Local properties will be overwritten by dbpedia values if there's a conflict
        mapp.mode = 'all'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value[uri_prop_completionDate], "")
        self.assertEqual(prop1.value[uri_prop_cinematography], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertIsInstance(prop1.value[uri_prop_starring], list)
        self.assertEqual(prop1.value["only_local"], "")
        self.assertEqual(prop1.value["Title"], "Jumanji")

        prop2 = results[1]
        self.assertEqual(prop2.value[uri_prop_director], "http://dbpedia.org/resource/Christopher_Nolan")
        self.assertEqual(prop2.value["only_local"], "")

        # Get all properties in DBPEDIA ONLY
        mapp.mode = 'all_retrieved'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value[uri_prop_completionDate], "")
        self.assertEqual(prop1.value[uri_prop_cinematography], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertIsInstance(prop1.value[uri_prop_starring], list)
        self.assertNotIn("only_local", prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value[uri_prop_completionDate], "")
        self.assertEqual(prop2.value[uri_prop_director], "http://dbpedia.org/resource/Christopher_Nolan")
        self.assertIsInstance(prop2.value[uri_prop_starring], list)
        self.assertNotIn("only_local", prop2.value)

        # Get all properties in LOCAL with value from DBPEDIA
        mapp.mode = 'original_retrieved'
        results = mapp.get_properties(self.raw_source)

        prop1 = results[0]
        self.assertEqual(prop1.value[uri_prop_cinematography], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertEqual(prop1.value["only_local"], '')
        self.assertNotIn(uri_prop_starring, prop1.value)

        prop2 = results[1]
        self.assertEqual(prop2.value["Title"], '')
        self.assertEqual(prop2.value["Budget_source"], '')
        self.assertEqual(prop2.value["only_local"], '')

        # # additional_filter Query
        # mapp = DBPediaMappingTechnique('Film', 'EN', 'Title', additional_filters={'budget': 'Budget_source',
        #                                                                           'wikiPageID': 'wiki_id'},
        #                                prop_as_uri=True)
        # results = mapp.get_properties(self.raw_source)
        #
        # prop1 = results[0]
        # self.assertEqual(prop1.value[uri_prop_cinematography], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        #
        # prop2 = results[1]
        # self.assertEqual(prop2.value[uri_prop_director], "http://dbpedia.org/resource/John_Lasseter")

    def test_entity_doesnt_exists(self):
        with self.assertRaises(ValueError):
            DBPediaMappingTechnique("dbo:not_exists", "Title")


class TestPropertiesFromDataset(TestCase):
    def test_get_properties(self):
        raw_source = JSONFile(source_path)

        # mode is 'only_retrieved_evaluated' so all item with a blank value
        # will be discarded
        expected_1 = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995",
                      "Budget_source": "6.5E7", "wiki_id": "3700174", "runtime (m)": "104.0"}
        expected_2 = {"Title": "Inception", "Budget_source":  "1.6E8"}
        mapp = PropertiesFromDataset()
        results = mapp.get_properties(raw_source)

        prop1 = results[0]
        prop2 = results[1]
        self.assertEqual(prop1.value, expected_1)
        self.assertEqual(prop2.value, expected_2)

        # mode is 'all' so all item are fine, also those with
        # a blank value
        expected_1 = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995",
                      "Budget_source": "6.5E7", "cinematography": "", "only_local": "", "wiki_id": "3700174",
                      "runtime (m)": "104.0"}
        expected_2 = {"Title": "Inception", "Budget_source": "1.6E8", "only_local": ""}
        mapp.mode = 'all'
        results = mapp.get_properties(raw_source)

        prop1 = results[0]
        prop2 = results[1]
        self.assertEqual(prop1.value, expected_1)
        self.assertEqual(prop2.value, expected_2)

        # field_name_list with 'title' is passed, and mode is 'only_retrieved_evaluated',
        # so only the field 'title' will be retrieved
        expected_1 = {'Title': 'Jumanji'}
        expected_2 = {'Title': 'Inception'}
        mapp = PropertiesFromDataset(field_name_list=['Title'])
        results = mapp.get_properties(raw_source)

        prop1 = results[0]
        prop2 = results[1]
        self.assertEqual(prop1.value, expected_1)
        self.assertEqual(prop2.value, expected_2)

        # field_name_list with a nonexistent field for a raw content is passed,
        # so nothing will be retrieved for the content with the missing field
        expected_1 = {"wiki_id": "3700174"}
        expected_2 = {}
        mapp = PropertiesFromDataset(field_name_list=['wiki_id'])
        results = mapp.get_properties(raw_source)

        prop1 = results[0]
        prop2 = results[1]
        self.assertEqual(prop1.value, expected_1)
        self.assertEqual(prop2.value, expected_2)

        # field_name_list with a nonexistent field for all raw_contents,
        # so nothing will be retrieved
        expected_1 = {}
        expected_2 = {}
        mapp = PropertiesFromDataset(field_name_list=['ppppp'])
        results = mapp.get_properties(raw_source)

        prop1 = results[0]
        prop2 = results[1]
        self.assertEqual(prop1.value, expected_1)
        self.assertEqual(prop2.value, expected_2)

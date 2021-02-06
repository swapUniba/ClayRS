from unittest import TestCase

from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique


class TestDBPediaMappingTechnique(TestCase):
    def test_get_properties(self):

        raw_content = {"Title": "Jumanji", "Year": "1995", "Rated": "PG",
                       "Released": "15 Dec 1995", "Budget_source": "6.5E7",
                       "cinematography": "", "only_local": "", "wiki_id": "3700174"}

        mapp = DBPediaMappingTechnique('Film', 'EN', 'Title')

        # Get properties THAT HAVE VALUE in DBPEDIA ONLY
        prop = mapp.get_properties('1', raw_content)
        self.assertEqual(prop.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertNotIn("only_local", prop.value)

        # Get all properties in DBPEDIA + Get all properties IN LOCAL SOURCE
        mapp.mode = 'all'
        prop = mapp.get_properties('1', raw_content)
        self.assertEqual(prop.value["completion_date"], "")
        self.assertEqual(prop.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertEqual(prop.value["only_local"], "")
        self.assertEqual(prop.value["Title"], "Jumanji")

        # Get all properties in DBPEDIA ONLY
        mapp.mode = 'all_retrieved'
        prop = mapp.get_properties('1', raw_content)
        self.assertEqual(prop.value["completion_date"], "")
        self.assertEqual(prop.value["cinematography"], "http://dbpedia.org/resource/Thomas_E._Ackerman")
        self.assertNotIn("only_local", prop.value)

        # Get all properties in LOCAL with value from DBPEDIA
        mapp.mode = 'original_retrieved'
        prop = mapp.get_properties('1', raw_content)
        self.assertEqual(prop.value["cinematography"], 'http://dbpedia.org/resource/Thomas_E._Ackerman')
        self.assertEqual(prop.value["only_local"], '')

        # additional_filter Query
        mapp = DBPediaMappingTechnique('Film', 'EN', 'Title', additional_filters={'budget': 'Budget_source', 'wikiPageID': 'wiki_id'})
        prop = mapp.get_properties('1', raw_content)

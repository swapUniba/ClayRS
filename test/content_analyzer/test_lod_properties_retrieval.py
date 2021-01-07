from unittest import TestCase

from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique


class TestDBPediaMappingTechnique(TestCase):
    def test_get_properties(self):
        self.skipTest("need SPARQL")
        raw_content = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995",
                       "Runtime": "104 min",
                       "Genre": "Adventure, Family, Fantasy", "Director": "Joe Johnston",
                       "Writer": "Jonathan Hensleigh (screenplay by), Greg Taylor (screenplay by), "
                                 "Jim Strain (screenplay by), Greg Taylor (screen story by), Jim Strain ("
                                 "screen story by), Chris Van Allsburg (screen story by), "
                                 "Chris Van Allsburg (based on the book by)",
                       "Actors": "Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce",
                       "Plot": "After being trapped in a jungle board game for 26 years, "
                               "a Man-Child wins his release from the game. But, no sooner has he arrived that he "
                               "is forced to play again, and this time sets the creatures of "
                               "the jungle loose on the city. Now it is up to him to stop them.",
                       "Language": "English, French", "Country": "USA", "Awards": "4 wins & 9 nominations.",
                       "Poster": "https://m.media-amazon.com/images/M/MV5BZTk2ZmUwYmEtNTcwZS00YmMyLWFkYjMtNTRmZDA3YWExMjc2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
                       "Ratings": [{"Source": "Internet Movie Database", "Value": "6.9/10"},
                                   {"Source": "Rotten Tomatoes", "Value": "53%"}, {"Source": "Metacritic", "Value": "39/100"}],
                       "Metascore": "39", "imdbRating": "6.9", "imdbVotes": "260,909", "imdbID": "tt0113497", "Type": "movie",
                       "DVD": "25 Jan 2000", "BoxOffice": "N/A", "Production": "Sony Pictures Home Entertainment",
                       "Website": "N/A", "Response": "True"}

        mapp = DBPediaMappingTechnique('Film', 'EN', 'Title')
        prop = mapp.get_properties('1', raw_content)

        mapp.mode = 'all'
        prop = mapp.get_properties('1', raw_content)

        mapp.mode = 'all_retrieved'
        prop = mapp.get_properties('1', raw_content)

        mapp.mode = 'original_retrieved'
        prop = mapp.get_properties('1', raw_content)

        mapp.label_field = 'Genre'
        mapp.mode = 'only_retrieved_evaluated'
        prop = mapp.get_properties('1', raw_content)

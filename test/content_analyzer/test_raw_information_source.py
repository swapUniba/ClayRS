from unittest import TestCase

from orange_cb_recsys.content_analyzer.raw_information_source import SQLDatabase, CSVFile, JSONFile


class TestSQLDatabase(TestCase):

    def test_iter(self):
        self.skipTest("FIX TEST")
        sql = SQLDatabase('localhost', 'root', 'password', 'prova', 'tabella')
        my_iter = iter(sql)
        d1 = {'campo1': 'Francesco', 'campo2': 'Benedetti', 'campo3': 'Polignano'}
        d2 = {'campo1': 'Mario', 'campo2': 'Rossi', 'campo3': 'Roma'}
        d3 = {'campo1': 'Gigio', 'campo2': 'Donnarumma', 'campo3': 'Milano'}
        self.assertDictEqual(next(my_iter), d1)
        self.assertDictEqual(next(my_iter), d2)
        self.assertDictEqual(next(my_iter), d3)


class TestCSVFile(TestCase):

    def test_iter(self):
        filepath = '../../datasets/movies_info_reduced.csv'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/movies_info_reduced.csv'

        csv = CSVFile(filepath)
        my_iter = iter(csv)
        d1 = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995", "Runtime": "104 min",
              "Genre": "Adventure, Family, Fantasy", "Director": "Joe Johnston",
              "Writer": "Jonathan Hensleigh (screenplay by), Greg Taylor (screenplay by), Jim Strain (screenplay by), Greg Taylor (screen story by), Jim Strain (screen story by), Chris Van Allsburg (screen story by), Chris Van Allsburg (based on the book by)",
              "Actors": "Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce",
              "Plot": "After being trapped in a jungle board game for 26 years, a Man-Child wins his release from the game. But, no sooner has he arrived that he is forced to play again, and this time sets the creatures of the jungle loose on the city. Now it is up to him to stop them.",
              "Language": "English, French", "Country": "USA", "Awards": "4 wins & 9 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BZTk2ZmUwYmEtNTcwZS00YmMyLWFkYjMtNTRmZDA3YWExMjc2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
              "Metascore": "39", "imdbRating": "6.9", "imdbVotes": "260,909", "imdbID": "tt0113497", "Type": "movie",
              "DVD": "25 Jan 2000", "BoxOffice": "N/A", "Production": "Sony Pictures Home Entertainment",
              "Website": "N/A",
              "Response": "True"}
        d2 = {"Title": "Grumpier Old Men", "Year": "1995", "Rated": "PG-13", "Released": "22 Dec 1995",
              "Runtime": "101 min",
              "Genre": "Comedy, Romance", "Director": "Howard Deutch",
              "Writer": "Mark Steven Johnson (characters), Mark Steven Johnson",
              "Actors": "Walter Matthau, Jack Lemmon, Sophia Loren, Ann-Margret",
              "Plot": "Things don't seem to change much in Wabasha County: Max and John are still fighting after 35 years, Grandpa still drinks, smokes, and chases women , and nobody's been able to catch the fabled \"Catfish Hunter\", a gigantic catfish that actually smiles at fishermen who try to snare it. Six months ago John married the new girl in town (Ariel), and people begin to suspect that Max might be missing something similar in his life. The only joy Max claims is left in his life is fishing, but that might change with the new owner of the bait shop.",
              "Language": "English, Italian, German", "Country": "USA", "Awards": "2 wins & 2 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BMjQxM2YyNjMtZjUxYy00OGYyLTg0MmQtNGE2YzNjYmUyZTY1XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
              "Metascore": "46", "imdbRating": "6.6", "imdbVotes": "21,823", "imdbID": "tt0113228", "Type": "movie",
              "DVD": "18 Nov 1997", "BoxOffice": "N/A", "Production": "Warner Home Video", "Website": "N/A",
              "Response": "True"}
        d3 = {"Title": "Toy Story", "Year": "1995", "Rated": "G", "Released": "22 Nov 1995", "Runtime": "81 min",
              "Genre": "Animation, Adventure, Comedy, Family, Fantasy", "Director": "John Lasseter",
              "Writer": "John Lasseter (original story by), Pete Docter (original story by), Andrew Stanton (original story by), Joe Ranft (original story by), Joss Whedon (screenplay by), Andrew Stanton (screenplay by), Joel Cohen (screenplay by), Alec Sokolow (screenplay by)",
              "Actors": "Tom Hanks, Tim Allen, Don Rickles, Jim Varney",
              "Plot": "A little boy named Andy loves to be in his room, playing with his toys, especially his doll named \"Woody\". But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life (as a toy) good. However, he must worry about Andy's family moving, and what Woody does not know is about Andy's birthday party. Woody does not realize that Andy's mother gave him an action figure known as Buzz Lightyear, who does not believe that he is a toy, and quickly becomes Andy's new favorite toy. Woody, who is now consumed with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.",
              "Language": "English", "Country": "USA",
              "Awards": "Nominated for 3 Oscars. Another 23 wins & 17 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg",
              "Metascore": "95", "imdbRating": "8.3", "imdbVotes": "761,649", "imdbID": "tt0114709", "Type": "movie",
              "DVD": "20 Mar 2001", "BoxOffice": "N/A", "Production": "Buena Vista",
              "Website": "http://www.disney.com/ToyStory", "Response": "True"}

        self.assertDictEqual(next(my_iter), d1)
        self.assertDictEqual(next(my_iter), d2)
        self.assertDictEqual(next(my_iter), d3)


class TestJSONFile(TestCase):

    def test_iter(self):
        filepath = '../../datasets/movies_info_reduced.json'
        try:
            with open(filepath):
                pass
        except FileNotFoundError:
            filepath = 'datasets/movies_info_reduced.json'

        csv = JSONFile(filepath)
        my_iter = iter(csv)
        d1 = {"Title": "Jumanji", "Year": "1995", "Rated": "PG", "Released": "15 Dec 1995", "Runtime": "104 min",
              "Genre": "Adventure, Family, Fantasy", "Director": "Joe Johnston",
              "Writer": "Jonathan Hensleigh (screenplay by), Greg Taylor (screenplay by), Jim Strain (screenplay by), Greg Taylor (screen story by), Jim Strain (screen story by), Chris Van Allsburg (screen story by), Chris Van Allsburg (based on the book by)",
              "Actors": "Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce",
              "Plot": "After being trapped in a jungle board game for 26 years, a Man-Child wins his release from the game. But, no sooner has he arrived that he is forced to play again, and this time sets the creatures of the jungle loose on the city. Now it is up to him to stop them.",
              "Language": "English, French", "Country": "USA", "Awards": "4 wins & 9 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BZTk2ZmUwYmEtNTcwZS00YmMyLWFkYjMtNTRmZDA3YWExMjc2XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
              "Ratings": [{"Source": "Internet Movie Database", "Value": "6.9/10"},
                          {"Source": "Rotten Tomatoes", "Value": "53%"}, {"Source": "Metacritic", "Value": "39/100"}],
              "Metascore": "39", "imdbRating": "6.9", "imdbVotes": "260,909", "imdbID": "tt0113497", "Type": "movie",
              "DVD": "25 Jan 2000", "BoxOffice": "N/A", "Production": "Sony Pictures Home Entertainment",
              "Website": "N/A",
              "Response": "True"}
        d2 = {"Title": "Grumpier Old Men", "Year": "1995", "Rated": "PG-13", "Released": "22 Dec 1995",
              "Runtime": "101 min",
              "Genre": "Comedy, Romance", "Director": "Howard Deutch",
              "Writer": "Mark Steven Johnson (characters), Mark Steven Johnson",
              "Actors": "Walter Matthau, Jack Lemmon, Sophia Loren, Ann-Margret",
              "Plot": "Things don't seem to change much in Wabasha County: Max and John are still fighting after 35 years, Grandpa still drinks, smokes, and chases women , and nobody's been able to catch the fabled \"Catfish Hunter\", a gigantic catfish that actually smiles at fishermen who try to snare it. Six months ago John married the new girl in town (Ariel), and people begin to suspect that Max might be missing something similar in his life. The only joy Max claims is left in his life is fishing, but that might change with the new owner of the bait shop.",
              "Language": "English, Italian, German", "Country": "USA", "Awards": "2 wins & 2 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BMjQxM2YyNjMtZjUxYy00OGYyLTg0MmQtNGE2YzNjYmUyZTY1XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_SX300.jpg",
              "Ratings": [{"Source": "Internet Movie Database", "Value": "6.6/10"},
                          {"Source": "Rotten Tomatoes", "Value": "17%"}, {"Source": "Metacritic", "Value": "46/100"}],
              "Metascore": "46", "imdbRating": "6.6", "imdbVotes": "21,823", "imdbID": "tt0113228", "Type": "movie",
              "DVD": "18 Nov 1997", "BoxOffice": "N/A", "Production": "Warner Home Video", "Website": "N/A",
              "Response": "True"}
        d3 = {"Title": "Toy Story", "Year": "1995", "Rated": "G", "Released": "22 Nov 1995", "Runtime": "81 min",
              "Genre": "Animation, Adventure, Comedy, Family, Fantasy", "Director": "John Lasseter",
              "Writer": "John Lasseter (original story by), Pete Docter (original story by), Andrew Stanton (original story by), Joe Ranft (original story by), Joss Whedon (screenplay by), Andrew Stanton (screenplay by), Joel Cohen (screenplay by), Alec Sokolow (screenplay by)",
              "Actors": "Tom Hanks, Tim Allen, Don Rickles, Jim Varney",
              "Plot": "A little boy named Andy loves to be in his room, playing with his toys, especially his doll named \"Woody\". But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life (as a toy) good. However, he must worry about Andy's family moving, and what Woody does not know is about Andy's birthday party. Woody does not realize that Andy's mother gave him an action figure known as Buzz Lightyear, who does not believe that he is a toy, and quickly becomes Andy's new favorite toy. Woody, who is now consumed with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.",
              "Language": "English", "Country": "USA",
              "Awards": "Nominated for 3 Oscars. Another 23 wins & 17 nominations.",
              "Poster": "https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg",
              "Ratings": [{"Source": "Internet Movie Database", "Value": "8.3/10"},
                          {"Source": "Rotten Tomatoes", "Value": "100%"}, {"Source": "Metacritic", "Value": "95/100"}],
              "Metascore": "95", "imdbRating": "8.3", "imdbVotes": "761,649", "imdbID": "tt0114709", "Type": "movie",
              "DVD": "20 Mar 2001", "BoxOffice": "N/A", "Production": "Buena Vista",
              "Website": "http://www.disney.com/ToyStory", "Response": "True"}

        self.assertDictEqual(next(my_iter), d1)
        self.assertDictEqual(next(my_iter), d2)
        self.assertDictEqual(next(my_iter), d3)

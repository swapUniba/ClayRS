import os
from unittest import TestCase

from clayrs.content_analyzer.raw_information_source import SQLDatabase, CSVFile, JSONFile, DATFile
from test import dir_test_files

json_file = os.path.join(dir_test_files, "movies_info_reduced.json")
csv_w_header = os.path.join(dir_test_files, 'movies_info_reduced.csv')
csv_no_header = os.path.join(dir_test_files, 'test_ratings', 'ratings_1591277020.csv')
dat_file = os.path.join(dir_test_files, 'users_70.dat')
tsv_file = os.path.join(dir_test_files, 'random_tsv.tsv')


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
        csv = CSVFile(csv_w_header)
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

    def test__len_w_header(self):
        csv = CSVFile(csv_w_header)

        self.assertEqual(3, len(csv))

    def test_iter_no_header(self):
        csv = CSVFile(csv_no_header, has_header=False)

        expected_row_1 = {'0': '01', '1': 'a', '2': '0.2333333333333333', '3': '1234567', '4': 'not so good',
                          '5': 'I expected more from this product', '6': '2.0'}
        expected_row_2 = {'0': '01', '1': 'b', '2': '0.8333333333333334', '3': '1234567', '4': 'perfect',
                          '5': 'I love this product', '6': '5.0'}
        expected_row_3 = {'0': '01', '1': 'c', '2': '0.8666666666666667', '3': '1234567', '4': 'awesome',
                          '5': 'The perfect gift for my darling', '6': '4.0'}
        expected_row_4 = {'0': '02', '1': 'a', '2': '-0.3666666666666667', '3': '1234567', '4': 'a disaster',
                          '5': 'Too much expensive ', '6': '1.0'}
        expected_row_5 = {'0': '02', '1': 'c', '2': '0.6', '3': '1234567', '4': 'really good',
                          '5': 'A good compromise', '6': '3.5'}
        expected_row_6 = {'0': '03', '1': 'b', '2': '0.6666666666666666', '3': '1234567', '4': 'Awesome',
                          '5': '', '6': '5.0'}

        csv_iterator = iter(csv)

        result_row_1 = next(csv_iterator)
        result_row_2 = next(csv_iterator)
        result_row_3 = next(csv_iterator)
        result_row_4 = next(csv_iterator)
        result_row_5 = next(csv_iterator)
        result_row_6 = next(csv_iterator)

        with self.assertRaises(StopIteration):
            next(csv_iterator)

        self.assertDictEqual(expected_row_1, result_row_1)
        self.assertDictEqual(expected_row_2, result_row_2)
        self.assertDictEqual(expected_row_3, result_row_3)
        self.assertDictEqual(expected_row_4, result_row_4)
        self.assertDictEqual(expected_row_5, result_row_5)
        self.assertDictEqual(expected_row_6, result_row_6)

    def test__len_no_header(self):
        csv = CSVFile(csv_no_header, has_header=False)

        self.assertEqual(6, len(csv))

    def test_iter_tsv(self):

        tsv = CSVFile(tsv_file, has_header=False, separator='\t')

        expected_row_1 = {'0': 'listen', '1': 'improve', '2': 'differ'}
        expected_row_2 = {'0': 'visitor', '1': 'meant', '2': 'kind'}
        expected_row_3 = {'0': 'basis', '1': 'climb', '2': 'honor'}
        expected_row_4 = {'0': 'simple', '1': 'vote', '2': 'closer'}
        expected_row_5 = {'0': 'blind', '1': 'finger', '2': 'pencil'}
        expected_row_6 = {'0': 'clock', '1': 'energy', '2': 'shape'}

        tsv_iterator = iter(tsv)

        result_row_1 = next(tsv_iterator)
        result_row_2 = next(tsv_iterator)
        result_row_3 = next(tsv_iterator)
        result_row_4 = next(tsv_iterator)
        result_row_5 = next(tsv_iterator)
        result_row_6 = next(tsv_iterator)

        with self.assertRaises(StopIteration):
            next(tsv_iterator)

        self.assertDictEqual(expected_row_1, result_row_1)
        self.assertDictEqual(expected_row_2, result_row_2)
        self.assertDictEqual(expected_row_3, result_row_3)
        self.assertDictEqual(expected_row_4, result_row_4)
        self.assertDictEqual(expected_row_5, result_row_5)
        self.assertDictEqual(expected_row_6, result_row_6)

        self.assertEqual(6, len(tsv))


class TestJSONFile(TestCase):

    def test_iter(self):

        js = JSONFile(json_file)
        my_iter = iter(js)
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

    def test__len(self):

        dat = JSONFile(json_file)
        self.assertEqual(20, len(dat))


class TestDATFile(TestCase):
    def test_iter(self):

        dat = DATFile(dat_file)
        my_iter = iter(dat)

        expected = [
            {'0': '1', '1': 'F', '2': '1', '3': '10', '4': '48067'},
            {'0': '2', '1': 'M', '2': '56', '3': '16', '4': '70072'},
            {'0': '3', '1': 'M', '2': '25', '3': '15', '4': '55117'}
        ]
        for line in expected:
            dat1 = next(my_iter)
            self.assertEqual(line, dat1)

    def test__len(self):

        dat = DATFile(dat_file)
        self.assertEqual(70, len(dat))

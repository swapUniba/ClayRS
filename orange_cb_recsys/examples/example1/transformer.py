import lzma
import pandas as pd

# id_vecchio - id_nuovo con query sul titolo
from orange_cb_recsys.content_analyzer.raw_information_source import DATFile, JSONFile, CSVFile

movies_info_filename = '../../../datasets/movies_info.json'
movies_filename = '../../../datasets/ml-1m/movies.csv'

ratings_filename = '../../../datasets/ml-1m/ratings.dat'
new_ratings_filename = '../../../datasets/new_ratings_full.csv'

movies_info = JSONFile(movies_info_filename)
movies = CSVFile(movies_filename)
ratings = DATFile(ratings_filename)

dict1 = {}
for film1 in movies_info:
    print('imdbID: {} |Title: {} ({})'.format(film1['imdbID'], film1['Title'], film1['Year']))
    dict1['{} ({})'.format(film1['Title'], film1['Year'])] = film1['imdbID']

dict2 = {}
for film2 in movies:
    print('movieId: {} |Title: {}'.format(film2['movieId'], film2['title']))
    dict2['{}'.format(film2['title'])] = film2['movieId']

dict3 = {}
for k in dict1.keys():
    if k in dict2.keys():
        dict3[dict2[k]] = dict1[k]

print(dict3)

print(len(dict3))

print('#####################################################################################')

new_ratings = pd.DataFrame()
for x, rating in enumerate(ratings):
    print('#{} : {}'.format(x, rating))
    user_id = rating['0']
    item_id = rating['1']
    score = rating['2']
    timestamp = int(rating['3'])
    if item_id in dict3.keys():
        new_ratings = new_ratings.append(pd.DataFrame({'user_id': [user_id],
                                                       'item_id': [dict3[item_id]],
                                                       'score': [score],
                                                       'timestamp': [timestamp]
                                                       }), ignore_index=True)
print(new_ratings)

new_ratings.to_csv(new_ratings_filename, index=False, header=True)
print('file saved in {}'.format(ratings_filename))

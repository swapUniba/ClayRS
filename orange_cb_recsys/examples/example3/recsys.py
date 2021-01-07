from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile
from orange_cb_recsys.evaluation import FNMeasure, Precision, RankingAlgEvalModel
from orange_cb_recsys.evaluation.prediction_metrics import RMSE, MAE
from orange_cb_recsys.recsys import NXPageRank
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.utils.feature_selection import NXFSPageRank
import pandas as pd

ratings_filename = '../../../datasets/examples/new_ratings.csv'
item_contents_dir = '../../../contents/examples/ex_3/movies_1600698490.340774'

ratings_import = RatingsImporter(
    source=CSVFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(field_name='points', processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp'
).import_ratings()


full_graph = NXFullGraph(
    source_frame=ratings_import,
    user_contents_dir=None,
    item_contents_dir=item_contents_dir,
    user_exogenous_properties=None,
    item_exogenous_properties=['film_director', 'starring', 'producer']
)

rank = NXPageRank(graph=full_graph).predict(
    user_id='1',
    ratings=ratings_import,
    recs_number=10,
)

rank_pd = pd.DataFrame({ 'from_id': ['1' for x in rank.keys()],
                         'to_id': [x for x in rank.keys()],
                         'rating': [x for x in rank.values()]})

truth_rank = ratings_import[ratings_import['from_id'] == '1']
truth_rank = truth_rank.rename(columns = {'score': 'rating'}, inplace = False)
print(truth_rank)

print(rank_pd)

print('RMSE: {}'.format(RMSE().perform(rank_pd, truth_rank)))
print('MAE: {}'.format(MAE().perform(rank_pd, truth_rank)))


rank = NXPageRank(graph=full_graph).predict(
    user_id='1',
    ratings=ratings_import,
    recs_number=10,
    feature_selection_algorithm=NXFSPageRank()
)


rank_pd = pd.DataFrame({ 'from_id': ['1' for x in rank.keys()],
                         'to_id': [x for x in rank.keys()],
                         'rating': [x for x in rank.values()]})

print(rank_pd)

print('RMSE: {}'.format(RMSE().perform(rank_pd, truth_rank)))
print('MAE: {}'.format(MAE().perform(rank_pd, truth_rank)))



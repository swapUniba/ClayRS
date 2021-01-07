from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, KFoldPartitioning, NDCG, ReportEvalModel, FNMeasure, \
    Correlation, Precision, Recall
from orange_cb_recsys.evaluation.prediction_metrics import MAE
from orange_cb_recsys.recsys import CentroidVector, CosineSimilarity, RecSysConfig, RecSys
import pandas as pd

ratings_filename = '../../../datasets/examples/new_ratings.csv'

items_ca_dir = '../../../contents/examples/ex_1/movies_1600355466.500496'

users_ca_dir = '../../../contents/examples/ex_1/users_1600355755.1935306'

ratings_importer = RatingsImporter(
    source=CSVFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(
        field_name='points',
        processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()

print(ratings_frame)


centroid_config = CentroidVector(
    item_field='Plot',
    field_representation='0',
    similarity=CosineSimilarity()
)


centroid_recsys_config = RecSysConfig(
    users_directory=users_ca_dir,
    items_directory=items_ca_dir,
    ranking_algorithm=centroid_config,
    rating_frame=ratings_frame
)


centroid_recommender = RecSys(
    config=centroid_recsys_config
)

rank: pd.DataFrame = centroid_recommender.fit_ranking(
    user_id='1',
    recs_number=5
)

print(rank)

rank.to_csv('out_ex_1.csv', index=False)

evaluation_centroid = RankingAlgEvalModel(
    config=centroid_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[FNMeasure(n=2), Precision(), Recall()]
)

results = evaluation_centroid.fit()

print(results)


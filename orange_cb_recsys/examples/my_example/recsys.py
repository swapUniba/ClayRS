from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig, RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, NDCG, FNMeasure, KFoldPartitioning, GiniIndex, DeltaGap, \
    ReportEvalModel
from orange_cb_recsys.recsys import ClassifierRecommender, RecSysConfig, RecSys

ratings_filename = '../../../datasets/ratings_example.json'
items_ca_dir = '../../../orange_cb_recsys/movie_dir1605298315.4501655'
users_ca_dir = '../../../datasets/examples/users_dir'

stars_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))

ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[stars_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()
print(ratings_frame)


original_classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory=users_ca_dir,
    items_directory=items_ca_dir,
    ranking_algorithm=original_classifier_config,
    rating_frame=ratings_frame
)

classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=15
)

print('original classifier')
print(rank)

new_classifier_config = ClassifierRecommender(
    _item_fields=['Plot', 'Genre'],
    _fields_representations={'Plot': ['0', '1'],
                             'Genre': ['0']},
    classifier='random_forest',
    classifier_parameters={
        'random_state': 42,
        'n_estimators': 400
    }
)

new_classifier_recsys_config = RecSysConfig(
    users_directory=users_ca_dir,
    items_directory=items_ca_dir,
    ranking_algorithm=new_classifier_config,
    rating_frame=ratings_frame
)

new_classifier_recommender = RecSys(
    config=new_classifier_recsys_config
)

new_rank = new_classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=15
)

print('new classifier')
print(new_rank)
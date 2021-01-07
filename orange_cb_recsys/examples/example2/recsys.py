from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig, RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, NDCG, FNMeasure, KFoldPartitioning, GiniIndex, DeltaGap, \
    ReportEvalModel
from orange_cb_recsys.recsys import ClassifierRecommender, RecSysConfig, RecSys

ratings_filename = '../../../datasets/examples/new_ratings.csv'
items_ca_dir = '../../../contents/examples/ex_2/movies_1600361344.090805'
users_ca_dir = '../../../contents/examples/ex_2/users_1600369975.663019'

# solo esempio, non presente nel dataset
"""
title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)
"""

points_review_config = RatingsFieldConfig(
    field_name='points',
    processor=NumberNormalizer(min_=1, max_=5))

ratings_importer = RatingsImporter(
    source=CSVFile(ratings_filename),          #cambia
    rating_configs=[points_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()
print(ratings_frame)

tfidf_classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory=users_ca_dir,
    items_directory=items_ca_dir,
    ranking_algorithm=tfidf_classifier_config,
    rating_frame=ratings_frame
)

classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='1',
    recs_number=5
)

print(rank)  # non salvare

evaluation_classifier = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[NDCG(), FNMeasure(n=2)]
)

evaluation_classifier_fairness = ReportEvalModel(
    config=classifier_recsys_config,
    recs_number=5,
    metric_list=[GiniIndex(), DeltaGap(user_groups={'a': 0.2, 'b': 0.8})]
)

results = evaluation_classifier.fit()
print(results)
results = evaluation_classifier_fairness.fit()
print(results)

# aggiungi metriche fairness


wordemb_classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='1',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory=users_ca_dir,
    items_directory=items_ca_dir,
    ranking_algorithm=wordemb_classifier_config,
    rating_frame=ratings_frame
)


classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank = classifier_recommender.fit_ranking(
    user_id='1',
    recs_number=5
)

print(rank)  # non salvare

evaluation_classifier = RankingAlgEvalModel(
    config=classifier_recsys_config,
    partitioning=KFoldPartitioning(n_splits=4),
    metric_list=[NDCG(), FNMeasure(n=2), GiniIndex(), DeltaGap(user_groups={'a': 0.2, 'b': 0.8})]
)

evaluation_classifier_fairness = ReportEvalModel(
    config=classifier_recsys_config,
    recs_number=5,
    metric_list=[GiniIndex(), DeltaGap(user_groups={'a': 0.2, 'b': 0.8})]
)

results = evaluation_classifier.fit()
print(results)
results = evaluation_classifier_fairness.fit()
print(results)

# aggiungi metriche fairness

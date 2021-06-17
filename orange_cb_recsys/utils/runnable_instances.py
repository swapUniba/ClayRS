import lzma
import os
import pickle
from typing import Dict

from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig, RatingsImporter
from orange_cb_recsys.content_analyzer import FieldConfig, ExogenousConfig, UserAnalyzerConfig, ItemAnalyzerConfig
from orange_cb_recsys.content_analyzer.embedding_learner import GensimWord2Vec, GensimDoc2Vec, GensimFastText, \
    GensimLatentSemanticAnalysis, GensimRandomIndexing
from orange_cb_recsys.content_analyzer.field_content_production_techniques import BabelPyEntityLinking, WhooshTfIdf, \
    Centroid
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique, \
    PropertiesFromDataset
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile, SQLDatabase, DATFile
from orange_cb_recsys.evaluation import RankingAlgEvalModel, KFoldPartitioning, NDCG, FNMeasure, Precision, Recall, \
    MRR, GiniIndex, PopRecsCorrelation, LongTailDistr, CatalogCoverage, DeltaGap, PopRatioVsRecs, Novelty, Correlation,\
    Serendipity
from orange_cb_recsys.evaluation.eval_model import PredictionAlgEvalModel, ReportEvalModel
from orange_cb_recsys.evaluation.prediction_metrics import RMSE, MAE
from orange_cb_recsys.recsys import ClassifierRecommender, NXPageRank, IndexQuery, RecSysConfig, RecSys
from orange_cb_recsys.recsys.ranking_algorithms.classifier import KNN, RandomForest, SVM, LogReg, DecisionTree, \
    GaussianProcess
from orange_cb_recsys.utils.const import logger

import pathlib
current_path = os.path.dirname(pathlib.Path(__file__).parent.absolute())

""" 
Default dict for all implementation of the abstract classes, for different purpose, 
with an 'alias' as key and the 'class name' as value
You can use this to show all implemented class in the framework
If a class is added to the framework and is a 'runnable_instance', 
you must add to this dict using add_runnable_instance() function 
or you can add manually in this dict and call __serialize() function 
with no arguments to add it permanently and also show in this file
"""

runnable_instances = {
    "field_config": FieldConfig,
    "recsys_config": RecSysConfig,
    "exogenous_config": ExogenousConfig,
    "item_analyzer": ItemAnalyzerConfig,
    "user_analyzer": UserAnalyzerConfig,
    "ratings": RatingsImporter,
    "ratings_config": RatingsFieldConfig,
    "recsys": RecSys,
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
    "dat": DATFile,
    "index": IndexInterface,
    "babelpy": BabelPyEntityLinking,
    "nltk": NLTK,
    "whoosh_tf-idf": WhooshTfIdf,
    "word2vec": GensimWord2Vec,
    "doc2vec": GensimDoc2Vec,
    "fasttext": GensimFastText,
    "latent_semantic_analysis": GensimLatentSemanticAnalysis,
    "random_indexing": GensimRandomIndexing,
    "centroid": Centroid,
    "text_blob_sentiment": TextBlobSentimentAnalysis,
    "number_normalizer": NumberNormalizer,
    "sk_learn_tf-idf": SkLearnTfIdf,
    "dbpedia_mapping": DBPediaMappingTechnique,
    "properties_from_dataset": PropertiesFromDataset,
    "synset_frequency": SynsetDocumentFrequency,
    "classifier": ClassifierRecommender,
    "knn": KNN,
    "random_forest": RandomForest,
    "svm": SVM,
    "log_reg": LogReg,
    "decision_tree": DecisionTree,
    "gaussian_process": GaussianProcess,
    "nx_page_rank": NXPageRank,
    "index_query": IndexQuery,
    "ranking_alg_eval_model": RankingAlgEvalModel,
    "prediction_alg_eval_model": PredictionAlgEvalModel,
    "report_eval_model": ReportEvalModel,
    "k_fold": KFoldPartitioning,
    "precision": Precision,
    "recall": Recall,
    "mrr": MRR,
    "fnmeasure": FNMeasure,
    "gini_index": GiniIndex,
    "pop_recs_correlation": PopRecsCorrelation,
    "long_tail_distribution": LongTailDistr,
    "catalog_coverage": CatalogCoverage,
    "delta_gap": DeltaGap,
    "popularity_ratio_vs_recs": PopRatioVsRecs,
    "novelty": Novelty,
    "rmse": RMSE,
    "mae": MAE,
    "ndcg": NDCG,
    "correlation": Correlation,
    "serendipity": Serendipity
}

"""
This contains, for each alias a specific category
"""

categories = {
    "embedding": 'content_production',
    "babelpy": 'content_production',
    "whoosh_tf-idf": 'content_production',
    "search_index": 'content_production',
    "sk_learn_tf-idf": 'content_production',
    "synset_frequency": 'content_production',
    "text_blob_sentiment": 'rating_processor',
    "number_normalizer": 'rating_processor',
    "nltk": 'preprocessor',
}


def __serialize(r_i: Dict[str, object], label: str):
    logger.info("Serializing runnable_instances in utils dir",)

    path = '{}/{}.xz'.format(current_path, label)
    try:
        with lzma.open(path, "rb") as f:
            pass
    except FileNotFoundError:
        path = 'contents/{}.xz'.format(label)

    with lzma.open(path, 'wb') as f:
        pickle.dump(r_i, f)


def get(alias: str = None):
    logger.info("Loading runnable_instances")
    r_i = {}
    try:
        path = '{}/runnable_instances.xz'.format(current_path)
        try:
            with lzma.open(path, "rb") as f:
                pass
        except FileNotFoundError:
            path = 'contents/runnable_instances.xz'
        with lzma.open(path, "rb") as f:
            r_i = pickle.load(f)
    except FileNotFoundError:
        logger.info('runnable_instances not found, dict is empty')
    if alias is None:
        return r_i
    elif alias in r_i.keys():
        return r_i[alias]
    else:
        logger.info('runnable_instance with %s alias not found', alias)
        return None


def get_cat(category: str = None, alias: str = None):
    """category: {'rating_processor', 'content_production', 'preprocessor'}"""
    if category is not None and category not in ['rating_processor', 'content_production', 'preprocessor']:
        raise ValueError('category not found')
    logger.info("Loading runnable_instances")
    cat = {}
    try:
        path = '{}/categories.xz'.format(current_path)
        try:
            with lzma.open(path, "rb") as f:
                pass
        except FileNotFoundError:
            path = 'contents/categories.xz'
        with lzma.open(path, "rb") as f:
            cat = pickle.load(f)
    except FileNotFoundError:
        logger.info('runnable_instances not found, dict is empty')
    if alias is None:
        if category is None:
            return cat
        return [k for k in cat.keys() if cat[k] == category]
    elif alias in cat.keys() and category is None:
        return cat[alias]
    elif alias in cat.keys() and category:
        return cat[alias] == category
    logger.info('runnable_instance with %s alias not found', alias)
    return None


def add(alias: str, runnable_class: object, category: str = None):
    """category: {'rating_processor', 'content_production', 'preprocessor'}"""
    if category is not None and category not in ['rating_processor', 'content_production', 'preprocessor']:
        raise ValueError('category not found')
    r_i = get()
    cat = get_cat()

    if alias in r_i.keys():
        logger.info('alias %s already exist, runnable_instance not added', alias)
    else:
        if category is not None:       # and is not in the r_i dict obv
            cat[alias] = category
        r_i[alias] = runnable_class
        __serialize(r_i, 'runnable_instances')
        __serialize(cat, 'categories')
        logger.info('%s successfully added', alias)


def remove(alias: str):
    r_i = get()
    if alias not in r_i.keys():
        logger.info('alias %s does not exist, runnable_instance not removed', alias)
    else:
        r_i.pop(alias)
        remove_from_categories(alias)
        __serialize(r_i, 'runnable_instances')
        logger.info('alias %s successfully removed', alias)


def remove_from_categories(alias: str):
    cat = get_cat()
    if alias not in cat.keys():
        logger.info('alias %s does not have a category', alias)
    else:
        cat.pop(alias)
        __serialize(cat, 'categories')
        logger.info('alias %s category successfully removed', alias)


def show(categories: bool=False):
    if categories:
        cat = get_cat()
        for k in cat.keys():
            logger.info('< %s : %s >', k, str(cat[k]))
    else:
        r_i = get()
        for k in r_i.keys():
            logger.info('< %s : %s >', k, str(r_i[k]))


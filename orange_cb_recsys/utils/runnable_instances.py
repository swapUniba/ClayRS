import lzma
import os
import pickle
from typing import Dict

from orange_cb_recsys.content_analyzer.field_content_production_techniques import BabelPyEntityLinking, WhooshTfIdf, \
    BinaryFile, GensimDownloader, Centroid, EmbeddingTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.field_content_production_techniques.synset_document_frequency import \
    SynsetDocumentFrequency
from orange_cb_recsys.content_analyzer.field_content_production_techniques.tf_idf import SkLearnTfIdf
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import DBPediaMappingTechnique
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile, CSVFile, SQLDatabase, DATFile
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
    "json": JSONFile,
    "csv": CSVFile,
    "sql": SQLDatabase,
    "dat": DATFile,
    "index": IndexInterface,
    "babelpy": BabelPyEntityLinking,
    "nltk": NLTK,
    "whoosh_tf-idf": WhooshTfIdf,
    "binary_file": BinaryFile,
    "gensim_downloader": GensimDownloader,
    "centroid": Centroid,
    "embedding": EmbeddingTechnique,
    "text_blob_sentiment": TextBlobSentimentAnalysis,
    "number_normalizer": NumberNormalizer,
    "search_index": SearchIndexing,
    "sk_learn_tf-idf": SkLearnTfIdf,
    "dbpedia_mapping": DBPediaMappingTechnique,
    "synset_frequency": SynsetDocumentFrequency,
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



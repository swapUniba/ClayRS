from typing import List, Type
import lzma
import pickle
import os

from orange_cb_recsys.content_analyzer import FieldConfig, ExogenousConfig, ContentAnalyzer
from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.embeddings.embedding_source import EmbeddingSource
from orange_cb_recsys.content_analyzer.exogenous_properties_retrieval import ExogenousPropertiesRetrieval
from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    FieldContentProductionTechnique
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import RatingProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter

from orange_cb_recsys.recsys.content_based_algorithm.classifier.classifiers import Classifier
from orange_cb_recsys.recsys.content_based_algorithm.regressor.regressors import Regressor
from orange_cb_recsys.recsys.graph_based_algorithm.feature_selection.feature_selection import FeatureSelectionAlgorithm
from orange_cb_recsys.recsys.graphs.graph_metrics import GraphMetrics
from orange_cb_recsys.recsys.algorithm import Algorithm
from orange_cb_recsys.recsys.graphs.graph import Graph
from orange_cb_recsys.recsys.content_based_algorithm.centroid_vector.similarities import Similarity
from orange_cb_recsys.recsys.recsys import RecSys

from orange_cb_recsys.evaluation import EvalModel
from orange_cb_recsys.evaluation.eval_pipeline_modules.methodology import Methodology
from orange_cb_recsys.evaluation.partitioning_techniques.partitioning import Partitioning
from orange_cb_recsys.evaluation.metrics.metrics import Metric
from orange_cb_recsys.evaluation.eval_pipeline_modules.metric_evaluator import MetricCalculator
from orange_cb_recsys.evaluation.eval_pipeline_modules.partition_module import Split, PartitionModule

from orange_cb_recsys.utils.class_utils import get_all_implemented_classes
from orange_cb_recsys.utils.const import root_path

"""
List containing all the base classes in the framework
"""
base_classes: List[Type] = [
    FieldConfig, ExogenousConfig, ContentAnalyzerConfig, FieldContentProductionTechnique, InformationProcessor,
    InformationInterface, EmbeddingSource, CombiningTechnique, RatingProcessor, RawInformationSource,
    ExogenousPropertiesRetrieval, RatingsImporter, ContentAnalyzer,

    RecSys, Algorithm, Graph, Similarity, Classifier, Regressor, GraphMetrics, FeatureSelectionAlgorithm,

    EvalModel, Metric, MetricCalculator, Partitioning, Methodology, Split, PartitionModule
]


def serialize_classes():
    """
    Serializes all the available framework's base classes

    To do so, a dictionary is created from the base_classes list and it will be in the following form:

        {
        "recsys": RecSys,
        "algorithm": Algorithm,
        ...
        }

    So each item in the dictionary will be in this form:

        Name of the class lowercased: Related class

    All the implemented classes from the base_class list will be added, meaning that if a class is abstract but has
    any implemented subclass, said subclasses will be added to the dictionary.
    If a class isn't abstract and has subclasses, both the class itself and the subclasses will be added to the
    dictionary

    The dictionary will then be serialized in a file called 'classes.xz' and, if the file already exists, it will be
    overwritten
    """
    if os.path.isfile(os.path.join(root_path, 'orange_cb_recsys/classes.xz')):
        os.remove(os.path.join(root_path, 'orange_cb_recsys/classes.xz'))

    classes_dict = {}

    for base_cls in base_classes:
        classes = get_all_implemented_classes(base_cls)
        for cls in classes:
            classes_dict[cls.__name__.lower()] = cls

    with lzma.open((root_path + 'orange_cb_recsys/classes.xz'), 'wb') as f:
        pickle.dump(classes_dict, f)


def get_classes():
    """
    Retrieves the implemented classes dictionary described in the serialize_classes function documentation from the
    'classes.xz' file. If the file is not found, a FileNotFoundError exception is thrown
    """
    try:
        path = os.path.join(root_path, 'orange_cb_recsys/classes.xz')
        with lzma.open(path, "rb") as f:
            classes = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('File containing classes not found')

    return classes


if __name__ == "__main__":
    serialize_classes()

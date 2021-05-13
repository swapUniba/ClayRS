import json
import pickle
import re
import lzma
from typing import Dict
import os
import numpy as np

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, FieldConfig
from orange_cb_recsys.content_analyzer.content_representation.content import Content, StringField, \
    FeaturesBagField, EmbeddingField
from orange_cb_recsys.content_analyzer.field_content_production_techniques. \
    field_content_production_technique import CollectionBasedTechnique, SingleContentTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.content_analyzer.content_representation.representation_container import RepresentationContainer
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.id_merger import id_merger


class ContentAnalyzer:
    """
    Class to whom the control of the content analysis phase is delegated. It uses the data stored in the configuration
    file to create and serialize the contents the user wants to produce. It also checks that the configurations the
    user wants to run on the raw contents have unique ids (otherwise it would be impossible to refer to a particular
    field representation or exogenous representation)

    Args:
        config (ContentAnalyzerConfig): configuration for processing the item fields. This parameter provides
            the possibility of customizing the way in which the input data is processed.
    """

    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def __dataset_refactor(self):
        """
        In case one or more CollectionBasedTechniques are used, a preprocessing phase needs to be computed in order to
        modify the dataset to fit accordingly to the technique. N.B. this will be removed in a future version to
        improve dynamic behavior (coincidentally with the refactor of the field content production techniques)
        """
        for field_name in self.__config.get_field_name_list():
            for config in self.__config.get_configs_list(field_name):

                technique = config.content_technique
                if isinstance(technique, CollectionBasedTechnique):
                    logger.info("Creating collection for technique: %s on field %s, "
                                "representation: %s", technique, field_name, config.id)
                    technique.field_need_refactor = field_name
                    technique.pipeline_need_refactor = str(config)
                    technique.processor_list = config.preprocessing
                    technique.dataset_refactor(self.__config.source, self.__config.id)

    def __serialize_content(self, content: Content):
        """
        This method serializes a specific content in the output directory defined by the content analyzer config

        Args:
            content (Content): content instance that will be serialized
        """
        logger.info("Serializing content %s in %s", content.content_id, self.__config.output_directory)

        file_name = re.sub(r'[^\w\s]', '', content.content_id)
        path = os.path.join(self.__config.output_directory, file_name + '.xz')
        with lzma.open(path, 'wb') as f:
            pickle.dump(content, f)

    def fit(self):
        """
        Processes the creation of the contents and serializes the contents. This method starts the content production
        process and initializes everything that will be used to create said contents, their fields and their
        representations
        """
        # before starting the process, the content analyzer manin checks that there are no duplicate id cases
        # both in the field dictionary and in the exogenous representation list
        # this is done now and not recursively for each content during the creation process, in order to avoid starting
        # an operation that is going to fail
        try:
            self.__check_field_dict()
            self.__check_exogenous_representation_list()
        except ValueError as e:
            raise e

        output_path = self.__config.output_directory
        os.mkdir(output_path)

        indexer = None
        if self.__config.search_index:
            index_path = os.path.join(output_path, 'search_index')
            indexer = IndexInterface(index_path)
            indexer.init_writing()

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)

        self.__dataset_refactor()
        contents_producer.set_indexer(indexer)
        i = 0
        for raw_content in self.__config.source:
            logger.info("Processing item %d", i)
            content = contents_producer.create_content(raw_content)
            self.__serialize_content(content)
            i += 1

        if self.__config.search_index:
            indexer.stop_writing()

        for field_name in self.__config.get_field_name_list():
            for config in self.__config.get_configs_list(field_name):
                technique = config.content_technique
                if isinstance(technique, CollectionBasedTechnique):
                    technique.delete_refactored()

    def __check_field_dict(self):
        """
        This function checks that there are no duplicate ids in the field_dict for a specific field_name.
        If this is not the case and a duplicate is found, a ValueError exception is thrown to warn the user.

        If the config id is None, the representation name will be unique by default (because the id of the
        representation that will be produced will be dynamically assigned during the content creation process).
        So any case where the id is None is not considered.
        """
        for field_name in self.__config.get_field_name_list():
            if len([config.id for config in self.__config.get_configs_list(field_name) if config.id is not None]) != \
               len(set(config.id for config in self.__config.get_configs_list(field_name) if config.id is not None)):
                raise ValueError("Each id for each field config of the field %s has to be unique" % field_name)

    def __check_exogenous_representation_list(self):
        """
        This function checks that there are no duplicate ids in the exogenous_representation_list.
        If this is not the case and a duplicate is found, a ValueError exception is thrown to warn the user.

        If the config id is None, the representation name will be unique by default (because the id of the
        representation that will be produced will be dynamically assigned during the content creation process)
        So any case where the id is None is not considered.
        """
        if len([config.id for config in self.__config.exogenous_representation_list if config.id is not None]) != \
           len(set(config.id for config in self.__config.exogenous_representation_list if config.id is not None)):
            raise ValueError("Each id for each exogenous config in the exogenous list has to be unique")

    def __str__(self):
        return "ContentAnalyzer"

    def __repr__(self):
        msg = "< " + "ContentAnalyzer: " + "" \
                                           "config = " + str(self.__config) + "; >"
        return msg


class ContentsProducer:
    """
    Singleton class which encapsulates the creation process of the items,
    The creation process is specified in the config parameter of ContentAnalyzer and
    is supposed to be the same for each item.
    """
    __instance = None

    @staticmethod
    def get_instance():
        """
        returns the singleton instance
        Returns:
            ContentsProducer: instance
        """
        # Static access method
        if ContentsProducer.__instance is None:
            ContentsProducer.__instance = ContentsProducer()
        return ContentsProducer.__instance

    def __init__(self):
        self.__config: ContentAnalyzerConfig = None
        self.__indexer = None
        # Virtually private constructor.
        if ContentsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        ContentsProducer.__instance = self

    def set_indexer(self, indexer: IndexInterface):
        self.__indexer = indexer

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def __create_field(self, raw_content: Dict, field_name: str, content_id: str):
        """
        Create a new field for the specified content

        Args:
            raw_content (Dict): Raw content for the new field
            field_name (str): Name of the new field
            content_id (str): Id of the content to which add the field

        Returns:
            field (ContentField)
        """
        field_data = raw_content[field_name]

        # two lists are instantiated, one for the configuration names (given by the user) and one for the field
        # representations. These lists will maintain the data for the field creation. This is done
        # because otherwise it would be necessary to append directly to the field. But in the ContentField class
        # the representations are kept as dataframes and appending to dataframes is computationally heavy
        field_representations_names = []
        field_representations = []

        for config in self.__config.get_configs_list(field_name):
            logger.info("Processing representation %d" % len(field_representations_names))
            field_representations_names.append(config.id)

            if isinstance(config.content_technique, CollectionBasedTechnique):
                field_representations.append(self.__create_representation_CBT(content_id, field_name, config))
            elif isinstance(config.content_technique, SingleContentTechnique):
                field_representations.append(self.__create_representation(field_data, config))
            elif isinstance(config.content_technique, SearchIndexing):
                self.__invoke_indexing_technique(field_name, field_data, config)
            elif config.content_technique is None:
                field_representations.append(self.__decode_field_data(field_data))

        # produce representations
        field = RepresentationContainer(field_representations, field_representations_names)

        return field

    def __decode_field_data(self, field_data: str):
        # Decode string into dict or list
        try:
            loaded = json.loads(field_data)
        except json.JSONDecodeError:
            try:
                # in case the dict is {'foo': 1} json expects {"foo": 1}
                reformatted_field_data = field_data.replace("\'", "\"")
                loaded = json.loads(reformatted_field_data)
            except json.JSONDecodeError:
                # if it has issues decoding we consider the data as str
                loaded = reformatted_field_data

        # if the decoded is a list, maybe it is an EmbeddingField repr
        if isinstance(loaded, list):
            arr = np.array(loaded)
            # if the array has only numbers then we consider it as a dense vector
            # else it is not and we consider the field data as a string
            if issubclass(arr.dtype.type, np.number):
                return EmbeddingField(arr)
            else:
                return StringField(field_data)

        # if the decoded is a dict, maybe it is a FeaturesBagField
        elif isinstance(loaded, dict):
            # if all values of the dict are numbers then we consider it as a bag of words
            # else it is not and we consider it as a string
            if len(loaded.values()) != 0 and \
                    all(isinstance(value, (float, int)) for value in loaded.values()):

                return FeaturesBagField(loaded)
            else:
                return StringField(field_data)

        # if the decoded is a string, then it is a StringField
        elif isinstance(loaded, str):
            return StringField(loaded)

    def __invoke_indexing_technique(self, field_name: str, field_data: str, config: FieldConfig):
        preprocessor_list = config.preprocessing
        processed_field_data = field_data
        for preprocessor in preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        config.content_technique.produce_content(field_name, str(config), processed_field_data, self.__indexer)

    @staticmethod
    def __create_representation_CBT(content_id: str, field_name: str, config: FieldConfig):
        return config.content_technique.produce_content(content_id, field_name)

    @staticmethod
    def __create_representation(field_data, config: FieldConfig):
        """
        Returns the specified representation for the specified field.

        Args:
            field_data: Raw data contained in the field
            config: FieldConfig object that contains the info regarding what to apply on the field (the production
                technique and the list of information processors)

        Returns:
            (Content)
        """
        preprocessor_list = config.preprocessing
        processed_field_data = field_data
        for preprocessor in preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        return config.content_technique.produce_content(processed_field_data)

    def create_content(self, raw_content: Dict):
        """
        Creates a content processing every field in the specified way.
        This method is iteratively invoked by the fit method.

        Args:
            raw_content (dict): Raw data from which the content will be created

        Returns:
            content (Content): an instance of content with his fields

        Raises:
            general Exception
        """

        if self.__config is None:
            raise Exception("You must set a config with set_config()")

        CONTENT_ID = "content_id"

        # two lists are instantiated, one for the configuration names (given by the user) and one for the exogenous
        # properties representations. These lists will maintain the data for the content creation. This is done
        # because otherwise it would be necessary to append directly to the content. But in the Content class
        # the representations are kept as dataframes and appending to dataframes is computationally heavy
        exo_config_names = []
        exo_properties = []

        for ex_config in self.__config.exogenous_representation_list:
            lod_properties = ex_config.exogenous_technique.get_properties(raw_content)
            exo_config_names.append(ex_config.id)
            exo_properties.append(lod_properties)

        # construct id from the list of the fields that compound id
        content_id = id_merger(raw_content, self.__config.id)
        content = Content(content_id, exogenous_rep_container=RepresentationContainer(exo_properties, exo_config_names))

        if self.__indexer is not None:
            self.__indexer.new_content()
            self.__indexer.new_field(CONTENT_ID, content_id)

        # produce
        for field_name in self.__config.get_field_name_list():
            logger.info("Processing field: %s", field_name)
            content.append_field(field_name, self.__create_field(raw_content, field_name, content_id))

        if self.__indexer is not None:
            content.index_document_id = self.__indexer.serialize_content()

        return content

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        msg = "< " + "ContentsProducer:" + "" \
                                           "config = " + str(self.__config) + " >"
        return msg

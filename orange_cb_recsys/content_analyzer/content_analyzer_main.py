from typing import Dict
import time
import os

from orange_cb_recsys.content_analyzer.config import ContentAnalyzerConfig, \
    FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.content_representation.content import Content, \
    RepresentedContentsRecap
from orange_cb_recsys.content_analyzer.content_representation.content_field import ContentField
from orange_cb_recsys.content_analyzer.field_content_production_techniques. \
    field_content_production_technique import \
    CollectionBasedTechnique, \
    SingleContentTechnique, SearchIndexing
from orange_cb_recsys.content_analyzer.memory_interfaces import IndexInterface
from orange_cb_recsys.utils.const import home_path, DEVELOPING, logger
from orange_cb_recsys.utils.id_merger import id_merger


class ContentAnalyzer:
    """
    Class to whom the control of the content analysis phase is delegated

    Args:
        config (ContentAnalyzerConfig):
            configuration for processing the item fields. This parameter provides the possibility
            of customizing the way in which the input data is processed.
    """

    def __init__(self, config: ContentAnalyzerConfig):
        self.__config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def __dataset_refactor(self):
        for field_name in self.__config.get_field_name_list():
            for pipeline in self.__config.get_pipeline_list(field_name):

                technique = pipeline.content_technique
                if isinstance(technique, CollectionBasedTechnique):
                    logger.info("Creating collection for technique: %s on field %s, "
                                "representation: %s", technique, field_name, pipeline)
                    technique.field_need_refactor = field_name
                    technique.pipeline_need_refactor = str(pipeline)
                    technique.processor_list = pipeline.preprocessor_list
                    technique.dataset_refactor(
                        self.__config.source, self.__config.id_field_name_list)

    def __config_recap(self):
        recap_list = [("Field: %s; representation id: %s: technique: %s",
                       field_name, str(pipeline), str(pipeline.content_technique))
                      for field_name in self.__config.get_field_name_list()
                      for pipeline in self.__config.get_pipeline_list(field_name)]

        return RepresentedContentsRecap(recap_list)

    def fit(self):
        """
        Processes the creation of the contents and serializes the contents
        """

        output_path = self.__config.output_directory
        if not DEVELOPING:
            output_path = os.path.join(home_path, 'contents', self.__config.output_directory)
        os.mkdir(output_path)

        indexer = None
        if self.__config.search_index:
            index_path = os.path.join(output_path, 'search_index')
            indexer = IndexInterface(index_path)
            indexer.init_writing()

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self.__config)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.init_writing()

        self.__dataset_refactor()
        contents_producer.set_indexer(indexer)
        i = 0
        for raw_content in self.__config.source:
            logger.info("Processing item %d", i)
            content = contents_producer.create_content(raw_content)
            content.serialize(output_path)
            i += 1

        if self.__config.search_index:
            indexer.stop_writing()

        for interface in interfaces:
            interface.stop_writing()

        for field_name in self.__config.get_field_name_list():
            for pipeline in self.__config.get_pipeline_list(field_name):
                technique = pipeline.content_technique
                if isinstance(technique, CollectionBasedTechnique):
                    technique.delete_refactored()

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

    def __get_timestamp(self, raw_content: Dict) -> str:
        """
        Search for timestamp as dataset field. If there isn't a field called 'timestamp', than
        the timestamp will be the one returned by the system.
        """
        timestamp = None
        if self.__config.content_type != "item":
            if "timestamp" in raw_content.keys():
                timestamp = raw_content["timestamp"]
            else:
                timestamp = time.time()

        return timestamp

    def __create_field(self, raw_content: Dict, field_name: str, content_id: str, timestamp: str):
        """
        Create a new field for the specified content

        Args:
            raw_content (Dict): Raw content for the new field
            field_name (str): Name of the new field
            content_id (str): Id of the content to which add the field
            timestamp (str)

        Returns:
            field (ContentField)
        """
        if isinstance(raw_content[field_name], list):
            timestamp = raw_content[field_name][1]
            field_data = raw_content[field_name][0]
        else:
            field_data = raw_content[field_name]

        # serialize for explanation
        memory_interface = self.__config.get_memory_interface(field_name)
        if memory_interface is not None:
            memory_interface.new_field(field_name, field_data)

        # produce representations
        field = ContentField(field_name, timestamp)

        for i, pipeline in enumerate(self.__config.get_pipeline_list(field_name)):
            logger.info("processing representation %d", i)
            if isinstance(pipeline.content_technique,
                          CollectionBasedTechnique):
                field.append(str(i),
                             self.__create_representation_CBT
                             (str(i), field_name, content_id, pipeline))

            elif isinstance(pipeline.content_technique, SingleContentTechnique):
                field.append(str(i), self.__create_representation(str(i), field_data, pipeline))
            elif isinstance(pipeline.content_technique, SearchIndexing):
                self.__invoke_indexing_technique(field_name, field_data, pipeline)
            elif pipeline.content_technique is None:
                field.append(str(i), field_data)

        return field

    def __invoke_indexing_technique(self, field_name: str, field_data: str,
                                    pipeline: FieldRepresentationPipeline):
        preprocessor_list = pipeline.preprocessor_list
        processed_field_data = field_data
        for preprocessor in preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        pipeline.content_technique.produce_content(field_name,
                                                   str(pipeline), processed_field_data,
                                                   self.__indexer)

    @staticmethod
    def __create_representation_CBT(field_representation_name: str,
                                    field_name: str, content_id: str,
                                    pipeline: FieldRepresentationPipeline):
        return pipeline.content_technique. \
            produce_content(field_representation_name, content_id, field_name)

    @staticmethod
    def __create_representation(field_representation_name: str, field_data,
                                pipeline: FieldRepresentationPipeline):
        """
        Returns the specified representation for the specified field.

        Args:
            field_representation_name: Name of the representation
            field_data: Raw data contained in the field
            pipeline: Preprocessing pipeline for the data

        Returns:
            (Content)
        """
        preprocessor_list = pipeline.preprocessor_list
        processed_field_data = field_data
        for preprocessor in preprocessor_list:
            processed_field_data = preprocessor.process(processed_field_data)

        return pipeline.content_technique. \
            produce_content(field_representation_name, processed_field_data)

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

        timestamp = self.__get_timestamp(raw_content)

        # construct id from the list of the fields that compound id
        content_id = id_merger(raw_content, self.__config.id_field_name_list)
        content = Content(content_id)

        for i, ex_retrieval in enumerate(self.__config.exogenous_properties_retrieval):
            lod_properties = ex_retrieval.get_properties(str(i), raw_content)
            content.append_exogenous_rep(str(i), lod_properties)

        if self.__indexer is not None:
            self.__indexer.new_content()
            self.__indexer.new_field(CONTENT_ID, content_id)

        interfaces = self.__config.get_interfaces()
        for interface in interfaces:
            interface.new_content()
            interface.new_field(CONTENT_ID, content_id)

        # produce
        for field_name in self.__config.get_field_name_list():
            logger.info("Processing field: %s", field_name)
            # search for timestamp override on specific field
            content.append(field_name,
                           self.__create_field
                           (raw_content, field_name, content_id, timestamp))

        if self.__indexer is not None:
            content.index_document_id = self.__indexer.serialize_content()

        for interface in interfaces:
            interface.serialize_content()

        return content

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        msg = "< " + "ContentsProducer:" + "" \
                                           "config = " + str(self.__config) + " >"
        return msg

from __future__ import annotations
import gc
import json
import pickle
import re
import lzma
import os
import shutil

from typing import List, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clayrs.content_analyzer.config import ContentAnalyzerConfig
    from clayrs.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface

from clayrs.content_analyzer.content_representation.content import Content, IndexField, ContentEncoder
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar
from clayrs.content_analyzer.utils.id_merger import id_merger


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
        self._config: ContentAnalyzerConfig = config

    def set_config(self, config: ContentAnalyzerConfig):
        self._config = config

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

        # creates the directory where the data will be serialized and overwrites it if it already exists
        output_path = self._config.output_directory
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        contents_producer = ContentsProducer.get_instance()
        contents_producer.set_config(self._config)
        created_contents = contents_producer.create_contents()

        if self._config.export_json:
            json_path = os.path.join(self._config.output_directory, 'contents.json')
            with open(json_path, "w") as data:
                json.dump(created_contents, data, cls=ContentEncoder, indent=4)

        with get_progbar(created_contents) as pbar:
            pbar.set_description("Serializing contents")

            for content in pbar:
                self.__serialize_content(content)

    def __serialize_content(self, content: Content):
        """
        This method serializes a specific content in the output directory defined by the content analyzer config
        Args:
            content (Content): content instance that will be serialized
        """

        file_name = re.sub(r'[^\w\s]', '', content.content_id)
        path = os.path.join(self._config.output_directory, file_name + '.xz')
        with lzma.open(path, 'wb') as f:
            pickle.dump(content, f, protocol=4)

    def __check_field_dict(self):
        """
        This function checks that there are no duplicate ids in the field_dict for a specific field_name.
        If this is not the case and a duplicate is found, a ValueError exception is thrown to warn the user.
        If the config id is None, the representation name will be kept as None even if it is not unique.
        So any case where the id is None is not considered.
        """
        for field_name in self._config.get_field_name_list():
            custom_id_list = [config.id for config in self._config.get_configs_list(field_name)
                              if config.id is not None]

            if len(custom_id_list) != len(set(custom_id_list)):
                raise ValueError("Each id for each field config of the field %s has to be unique" % field_name)

    def __check_exogenous_representation_list(self):
        """
        This function checks that there are no duplicate ids in the exogenous_representation_list.
        If this is not the case and a duplicate is found, a ValueError exception is thrown to warn the user.
        If the config id is None, the representation name will be kept as None even if it is not unique.
        So any case where the id is None is not considered
        """
        custom_id_list = [config.id for config in self._config.exogenous_representation_list if config.id is not None]
        if len(custom_id_list) != len(set(custom_id_list)):
            raise ValueError("Each id for each exogenous config in the exogenous list has to be unique")

    def __str__(self):
        return "ContentAnalyzer"

    def __repr__(self):
        return f'ContentAnalyzer(config={self._config})'


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
        self.__config: Optional[ContentAnalyzerConfig] = None
        # dictionary of memory interfaces defined in the FieldConfigs. The key is the directory of the memory interface
        # and the value is the memory interface itself (only one memory interface can be defined for each directory)
        # if a memory interface has an already defined directory, the memory interface associated to said directory
        # will be considered instead
        self.__memory_interfaces: Dict[InformationInterface] = {}
        # Virtually private constructor.
        if ContentsProducer.__instance is not None:
            raise Exception("This class is a singleton!")
        ContentsProducer.__instance = self

    def set_config(self, config: ContentAnalyzerConfig):
        self.__config = config

    def create_contents(self) -> List[Content]:
        """
        Creates the contents based on the information defined in the Content Analyzer's config
        Returns:
            contents_list (List[Content]): list of contents created by the method
        """
        if self.__config is None:
            raise Exception("You must set a config with set_config()")

        # will store the contents and is the variable that will be returned by the method
        contents_list = []

        for raw_content in self.__config.source:
            # construct id from the list of the fields that compound id
            content_id = id_merger(raw_content, self.__config.id)
            contents_list.append(Content(content_id))

        # two lists are instantiated, one for the configuration names (given by the user) and one for the exogenous
        # properties representations. These lists will maintain the data for the content creation. This is done
        # because otherwise it would be necessary to append directly to the content. But in the Content class
        # the representations are kept as dataframes and appending to dataframes is computationally heavy
        for ex_config in self.__config.exogenous_representation_list:
            lod_properties = ex_config.exogenous_technique.get_properties(self.__config.source)

            for i in range(len(contents_list)):
                contents_list[i].append_exogenous_representation(lod_properties[i], ex_config.id)

        # this dictionary will store any representation list that will be kept in one of the the index
        # the elements will be in the form:
        #   { memory_interface: {'Plot_0': [FieldRepr for content1, FieldRepr for content2, ...]}}
        # the 0 after the Plot field name is used to define the representation number associated with the Plot field
        # since it's possible to store multiple Plot fields in the index
        index_representations_dict = {}

        for field_name in self.__config.get_field_name_list():
            logger.info(f"   Processing field: {field_name}   ".center(50, '*'))

            for repr_number, field_config in enumerate(self.__config.get_configs_list(field_name)):

                # technique_result is a list of field representation produced by the content technique
                # each field repr in the list will refer to a content
                # technique_result[0] -> contents_list[0]
                technique_result = field_config.content_technique.produce_content(
                    field_name, field_config.preprocessing, field_config.postprocessing, self.__config.source)

                if field_config.memory_interface is not None:
                    memory_interface = field_config.memory_interface
                    # if the index for the directory in the config hasn't been defined yet in the contents producer,
                    # the index associated to the field config that is being processed is added to the
                    # contents producer's memory interfaces list, and will be used for the future field configs with
                    # an assigned memory interface that has the same directory.
                    # This means that only the index defined in the first FieldConfig that has one will actually be used
                    if memory_interface not in self.__memory_interfaces.values():
                        self.__memory_interfaces[memory_interface.directory] = memory_interface
                        index_representations_dict[memory_interface] = {}
                    else:
                        memory_interface = self.__memory_interfaces[memory_interface.directory]

                    if field_config.id is not None:
                        index_field_name = "{}#{}#{}".format(field_name, str(repr_number), field_config.id)
                    else:
                        index_field_name = "{}#{}".format(field_name, str(repr_number))

                    index_representations_dict[memory_interface][index_field_name] = technique_result

                    # in order to refer to the representation that will be stored in the index, an IndexField repr will
                    # be added to each content (and it will contain all the necessary information to retrieve the data
                    # from the index)
                    technique_result = [IndexField(index_field_name, i, memory_interface)
                                        for i in range(len(self.__config.source))]

                for i in range(len(contents_list)):
                    contents_list[i].append_field_representation(field_name, technique_result[i], field_config.id)

                del technique_result
                gc.collect()

        # after the contents creation process, the data to be indexed will be serialized inside of the memory interfaces
        # for each created content, a new entry in each index will be created
        # the entry will be in the following form: {"content_id": id, "Plot_0": "...", "Plot_1": "...", ...}
        if len(self.__memory_interfaces) != 0:
            for memory_interface in self.__memory_interfaces.values():
                memory_interface.init_writing(True)
                for i in range(0, len(contents_list)):
                    memory_interface.new_content()
                    memory_interface.new_field("content_id", contents_list[i].content_id)
                    for field_name in index_representations_dict[memory_interface].keys():
                        memory_interface.new_field(
                            field_name, str(index_representations_dict[memory_interface][field_name][i].value))
                    memory_interface.serialize_content()
                memory_interface.stop_writing()
            self.__memory_interfaces.clear()

        return contents_list

    def __str__(self):
        return "ContentsProducer"

    def __repr__(self):
        return f'ContentsProducer()'

from __future__ import annotations
import abc
import re
from abc import ABC
from typing import List, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
        FieldContentProductionTechnique
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor
    from clayrs.content_analyzer.exogenous_properties_retrieval import ExogenousPropertiesRetrieval
    from clayrs.content_analyzer.memory_interfaces.memory_interfaces import InformationInterface
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.information_processor.postprocessors import PostProcessor

from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    OriginalData


class FieldConfig:
    """
    Class that represents the configuration for a single representation of a field. The configuration of a single
    representation is defined by a `FieldContentProductionTechnique` (e.g. an `EmbeddingTechnique`) that will be applied
    to the pre-processed data of said field.

    To specify how to preprocess data, simply specify an `InformationProcessor` in the `preprocessing` parameter.
    Multiple `InformationProcessor` can be wrapped in a list: in this case, the field will be preprocessed by performing
    all operations inside the list sequentially.
    If preprocessing is not defined, no preprocessing operations will be done on the field data.

    Additionally, it is also possible to specify postprocessors with `PostProcessorConfig` in the `postprocessing` parameter.
    Multiple `PostProcessorConfig` can be wrapped in a list: in this case, the field will be postprocessed by performing
    all operations inside the list sequentially.
    If postprocessing is not defined, no postprocessing operations will be done on the field data.

    You can use the `id` parameter to assign a custom id for the representation: by doing so the user can freely refer
    to it by using the custom id given, rather than positional integers (which are given automatically by the
    framework).

    There is also a memory_interface attribute which allows to define a data structure where the representation will
    be serialized (e.g. an Index).

    Various configurations are possible depending on how the user wants to represent a particular field:

    * This will produce a field representation using the SkLearnTfIdf technique on the field data
      preprocessed by NLTK by performing stopwords removal, and the name of the produced representation will be
      'field_example'

    ```python
    FieldConfig(SkLearnTfIdf(), NLTK(stopwords_removal=True), id='field_example')
    ```

    * This will produce the same result as above but the id for the field representation defined by this config will
      be set by the ContentAnalyzer once it is being processed (0 integer if it's the first representation specified
      for the field, 1 if it's the second, etc.)

    ```python
    FieldConfig(SkLearnTfIdf(), NLTK())
    ```

    * This will produce a field representation using the SkLearnTfIdf technique on the field data
      preprocessed by NLTK by performing stopwords removal and postprocessed by performing PCA,
      and the name of the produced representation will be None since no id was defined for the post processor

    ```python
    FieldConfig(SkLearnTfIdf(), NLTK(stopwords_removal=True), PostProcessorConfig(PCA(100)), id='field_example')
    ```

    * This will produce the same result as above but the id for the field representation defined by this config will
      be "field_example_pca_100" obtained by concatenating the field config id and the postprocessor id

    ```python
    FieldConfig(SkLearnTfIdf(), NLTK(stopwords_removal=True), PostProcessorConfig(PCA(100), id="pca_100"), id='field_example')
    ```

    * This will produce two representations, one with id "field_example_pca_100" and one with id "field_example_pca_200",
      obtained by postprocessing the same representation (nltk + tfidf)

    ```python
    post_processor_list = [PostProcessorConfig(PCA(100), id="pca_100"), PostProcessorConfig(PCA(200), id="pca_200")]
    FieldConfig(SkLearnTfIdf(), NLTK(stopwords_removal=True), id='field_example')
    ```

    * This will produce a field representation using the SkLearnTfIdf technique on the field data without applying
      any preprocessing operation, but it will not be directly stored in the content, instead it will be
      stored in a index

    ```python
    FieldConfig(SkLearnTfIdf(), memory_interface=SearchIndex(/somedir))
    ```

    * In the following nothing will be done on the field data, it will be represented as is

    ```python
    FieldConfig()
    ```

    Args:
        content_technique: Technique that will be applied to the field in order to produce a complex representation
            of said field
        preprocessing: Single `InformationProcessor` object or a list of `InformationProcessor` objects that will be
            used preprocess field data before applying the `content_technique`
        memory_interface: complex structure where the content representation can be serialized (an Index for
            example)
        id: Custom id that can be used later by the user to easily refer to the representation generated by this config.
            IDs for a single field should be unique! And should only contain '_', '-' and alphanumeric characters
    """

    def __init__(self,
                 content_technique: FieldContentProductionTechnique = OriginalData(),
                 preprocessing: Union[InformationProcessor, List[InformationProcessor]] = None,
                 postprocessing: Union[PostProcessorConfig, List[PostProcessorConfig]] = None,
                 memory_interface: InformationInterface = None,
                 id: str = None):

        if preprocessing is None:
            preprocessing = []

        if postprocessing is None:
            postprocessing = []

        if id is not None:
            self._check_custom_id(id)

        self.__content_technique = content_technique
        self.__preprocessing = preprocessing
        self.__postprocessing = postprocessing
        self.__memory_interface = memory_interface
        self.__id = id

        if not isinstance(self.__preprocessing, list):
            self.__preprocessing = [self.__preprocessing]

        if not isinstance(self.__postprocessing, list):
            self.__postprocessing = [self.__postprocessing]

    @property
    def memory_interface(self):
        """
        Getter for the index associated to the field config
        """
        return self.__memory_interface

    @property
    def content_technique(self):
        """
        Getter for the field content production technique of the field
        """
        return self.__content_technique

    @property
    def preprocessing(self):
        """
        Getter for the list of preprocessor of the field config
        """
        return self.__preprocessing

    @property
    def postprocessing(self):
        """
        Getter for the list of postprocessor of the field config
        """
        return self.__postprocessing

    @property
    def id(self):
        """
        Getter for the id of the field config
        """
        return self.__id

    def _check_custom_id(self, id: str):
        if not re.match("^[A-Za-z0-9_-]+$", id):
            raise ValueError("The custom id {} is not valid!\n"
                             "A custom id can only have numbers, letters and '_' or '-'!".format(id))

    def __str__(self):
        return "FieldConfig"

    def __repr__(self):
        return f'FieldConfig(content_technique={self.__content_technique}, preprocessing={self.__preprocessing}, ' \
               f'memory_interface={self.__memory_interface}, id={self.__id})'


class PostProcessorConfig:
    """
    Class that represents the configuration for a single post-processed representation.

    The config allows you to specify a `PostProcessor` technique to use to refine content representations.
    W.r.t `FieldConfig` instances, a `PostProcessor` config is an argument to them.

    You can use the `id` parameter to assign a custom id for the representation: by doing so the user can freely refer
    to it by using the custom id given, rather than positional integers (which are given automatically by the
    framework).

    Note that, ids for the representations generated will be a concatenation of the field config id and postprocessor
    id, so for example if a `FieldConfig` with `id` parameter 'Plot' is specified and a `PostProcessorConfig` with `id`
    'PCA' is specified, then the final external id will be 'Plot_PCA'.

    * This will refine a field representation for a content by postprocessing it, said refinement will be named 'test'
    ```python
    PostProcessorConfig(SkLearnPCA(400), id='test')
    ```

    Refer to 'FieldConfig' documentation to see better example related to the usage of 'PostProcessorConfig' on
    produced representations.

    Args:
        postprocessor_technique: Technique or list of techniques which will be used to refine the produced content
            representation
        id: Custom id that can be used later by the user to easily refer to the representation generated by this config.
            IDs should be unique! And should only contain '_', '-' and alphanumeric characters
    """

    def __init__(self, postprocessor_technique: Union[PostProcessor, List[PostProcessor]] = None, id: str = None):

        if id is not None:
            self._check_custom_id(id)

        self.__postprocessor_technique = postprocessor_technique
        self.__id = id

        if self.__postprocessor_technique is None:
            self.__postprocessor_technique = []

        if not isinstance(self.__postprocessor_technique, list):
            self.__postprocessor_technique = [self.__postprocessor_technique]

    @property
    def postprocessor_technique(self):
        """
        Getter for the exogenous properties retrieval technique
        """
        return self.__postprocessor_technique

    @property
    def id(self):
        """
        Getter for the ExogenousConfig id
        """
        return self.__id

    def _check_custom_id(self, id: str):
        if not re.match("^[A-Za-z0-9_-]+$", id):
            raise ValueError("The custom id {} is not valid!\n"
                             "A custom id can only have numbers, letters and '_' or '-'!".format(id))

    def __str__(self):
        return "PostProcessorConfig"

    def __repr__(self):
        return f'PostProcessorConfig(postprocessor_technique={self.__postprocessor_technique}, ' \
               f'id={self.__id})'


class ExogenousConfig:
    """
    Class that represents the configuration for a single exogenous representation.

    The config allows the user to specify an `ExogenousPropertiesRetrieval` technique to use to expand each content.
    W.r.t `FieldConfig` objects, an `ExogenousConfig` does not refer to a particular field but to the whole content
    itself.

    You can use the `id` parameter to assign a custom id for the representation: by doing so the user can freely refer
    to it by using the custom id given, rather than positional integers (which are given automatically by the
    framework).

    * This will create an exogenous representation for the content by expanding it using DBPedia,
    said representation will be named 'test'
    ```python
    ExogenousConfig(DBPediaMappingTechnique('dbo:Film', 'Title', 'EN'), id='test')
    ```

    * Same as the example above, but since no custom id was assigned, the exogenous representation can be referred to
    only with an integer (0 if it's the first exogenous representation specified for the contents, 1 if it's the second,
    etc.)

    ```python
    ExogenousConfig(DBPediaMappingTechnique('dbo:Film', 'Title', 'EN'))
    ```

    Args:
        exogenous_technique: Technique which will be used to expand each content with data from external sources.
            An example would be the DBPediaMappingTechnique which allows to retrieve properties from DBPedia.
        id: Custom id that can be used later by the user to easily refer to the representation generated by this config.
            IDs for a single field should be unique! And should only contain '_', '-' and alphanumeric characters
    """

    def __init__(self, exogenous_technique: ExogenousPropertiesRetrieval, id: str = None):
        if id is not None:
            self._check_custom_id(id)

        self.__exogenous_technique = exogenous_technique
        self.__id = id

    @property
    def exogenous_technique(self):
        """
        Getter for the exogenous properties retrieval technique
        """
        return self.__exogenous_technique

    @property
    def id(self):
        """
        Getter for the ExogenousConfig id
        """
        return self.__id

    def _check_custom_id(self, id: str):
        if not re.match("^[A-Za-z0-9_-]+$", id):
            raise ValueError("The custom id {} is not valid!\n"
                             "A custom id can only have numbers, letters and '_' or '-'!".format(id))

    def __str__(self):
        return "ExogenousConfig"

    def __repr__(self):
        return f'ExogenousConfig(exogenous_technique={self.__exogenous_technique}, ' \
               f'id={self.__id})'


class ContentAnalyzerConfig(ABC):
    """
    Abstract class that represents the configuration for the content analyzer. The configuration specifies how the
    `Content Analyzer` needs to complexly represent contents, i.e. how to preprocess them and how to represent them

    Args:
        source: Raw data source wrapper which contains original information about contents to process
        id: Field of the raw source which represents each content uniquely.
        output_directory: Where contents complexly represented will be serialized
        field_dict: Dictionary object which contains, for each field of the raw source to process, a FieldConfig object
            (e.g. `{'plot': FieldConfig(SkLearnTfIdf(), 'genres': FieldConfig(WhooshTfIdf()))}`)
        exogenous_representation_list: List of `ExogenousTechnique` objects that will be used to expand each contents
            with data from external sources
        export_json: If set to True, contents complexly represented will be serialized in a human readable JSON, other
            than in a proprietary format of the framework
    """

    def __init__(self, source: RawInformationSource,
                 id: Union[str, List[str]],
                 output_directory: str,
                 field_dict: Dict[str, List[FieldConfig]] = None,
                 exogenous_representation_list: Union[ExogenousConfig, List[ExogenousConfig]] = None,
                 export_json: bool = False):
        if field_dict is None:
            field_dict = {}
        if exogenous_representation_list is None:
            exogenous_representation_list = []

        self.__source = source
        self.__id = id
        self.__output_directory = output_directory
        self.__field_dict = field_dict
        self.__exogenous_representation_list = exogenous_representation_list
        self.__export_json = export_json

        if not isinstance(self.__exogenous_representation_list, list):
            self.__exogenous_representation_list = [self.__exogenous_representation_list]

        if not isinstance(self.__id, list):
            self.__id = [self.__id]

    @property
    def output_directory(self):
        """
        Getter for the output directory where the produced contents will be stored
        """
        return self.__output_directory

    @property
    def id(self) -> List[str]:
        """
        Getter for the id that represents the ids of the produced contents
        """
        return self.__id

    @property
    def source(self) -> RawInformationSource:
        """
        Getter for the raw information source where the original contents are stored
        """
        return self.__source

    @property
    def exogenous_representation_list(self) -> List[ExogenousConfig]:
        """
        Getter for the exogenous_representation_list
        """
        return self.__exogenous_representation_list

    @property
    def export_json(self) -> bool:
        """
        Getter for the export_json parameter
        """
        return self.__export_json

    def get_configs_list(self, field_name: str) -> List[FieldConfig]:
        """
        Method which returns the list of all `FieldConfig` objects specified for the input `field_name` parameter

        Args:
            field_name: Name of the field for which the list of field configs will be retrieved

        Returns:
            List containing all `FieldConfig` objects specified for the input `field_name`
        """
        return [config for config in self.__field_dict[field_name]]

    def get_field_name_list(self) -> List[str]:
        """
        Method which returns a list containing all the fields of the raw source for which at least one `FieldConfig`
        object has been assigned (i.e. at least one complex representations is specified)

        Returns:
            List of all the fields of the raw source that must be complexly represented
        """
        return list(self.__field_dict.keys())

    def add_single_config(self, field_name: str, field_config: FieldConfig):
        """
        Method which adds a single complex representation for the `field_name` of the raw source

        Examples:

            * Represent field "Plot" of the raw source with a tf-idf technique using sklearn
            >>> import clayrs.content_analyzer as ca
            >>> movies_ca_config.add_single_config("Plot", FieldConfig(ca.SkLearnTfIdf()))

        Args:
            field_name: field name of the raw source which must be complexly represented
            field_config: `FieldConfig` specifying how to represent the field of the raw source
        """
        # If the field_name is not in the field_dict keys it means there is no list to append the FieldConfig to,
        # so a new list is instantiated
        if self.__field_dict.get(field_name) is not None:
            self.__field_dict[field_name].append(field_config)
        else:
            self.__field_dict[field_name] = list()
            self.__field_dict[field_name].append(field_config)

    def add_multiple_config(self, field_name: str, config_list: List[FieldConfig]):
        """
        Method which adds multiple complex representations for the `field_name` of the raw source

        Examples:

            * Represent preprocessed field "Plot" of the raw source with a tf-idf technique using sklearn and a word
            embedding technique using Word2Vec. For the latter, no preprocessing operation will be applied
            >>> import clayrs.content_analyzer as ca
            >>> movies_ca_config.add_multiple_config("Plot",
            >>>                                       [FieldConfig(ca.SkLearnTfIdf(),
            >>>                                                    preprocessing=ca.NLTK(stopwords_removal=True)),
            >>>
            >>>                                        FieldConfig(ca.WordEmbeddingTechnique(ca.GensimWord2Vec()))]

        Args:
            field_name: field name of the raw source which must be complexly represented
            config_list: List of `FieldConfig` objects specifying how to represent the field of the raw source
        """
        # If the field_name is not in the field_dict keys it means there is no list to append the FieldConfig to,
        # so a new list is instantiated
        if self.__field_dict.get(field_name) is not None:
            self.__field_dict[field_name].extend(config_list)
        else:
            self.__field_dict[field_name] = list()
            self.__field_dict[field_name].extend(config_list)

    def add_single_exogenous(self, exogenous_config: ExogenousConfig):
        """
        Method which adds a single exogenous representation which will be used to expand each content

        Examples:

            * Expand each content by using DBPedia as external source
            >>> import clayrs.content_analyzer as ca
            >>> movies_ca_config.add_single_exogenous(
            >>>     ca.ExogenousConfig(
            >>>         ca.DBPediaMappingTechnique('dbo:Film', 'Title', 'EN')
            >>>     )
            >>> )

        Args:
            exogenous_config: `ExogenousConfig` object specifying how to expand each content
        """
        self.__exogenous_representation_list.append(exogenous_config)

    def add_multiple_exogenous(self, config_list: List[ExogenousConfig]):
        """
        Method which adds multiple exogenous representations which will be used to expand each content

        Examples:

            * Expand each content by using DBPedia as external source and local dataset as external source
            >>> import clayrs.content_analyzer as ca
            >>> movies_ca_config.add_single_exogenous(
            >>>     [
            >>>         ca.ExogenousConfig(
            >>>             ca.DBPediaMappingTechnique('dbo:Film', 'Title', 'EN')
            >>>         ),
            >>>
            >>>         ca.ExogenousConfig(
            >>>             ca.PropertiesFromDataset(field_name_list=['director'])
            >>>         ),
            >>>     ]
            >>> )

        Args:
            config_list: List containing `ExogenousConfig` objects specifying how to expand each content
        """
        self.__exogenous_representation_list.extend(config_list)

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class UserAnalyzerConfig(ContentAnalyzerConfig):
    """
    Class that represents the configuration for the content analyzer. The configuration specifies how the
    `Content Analyzer` needs to complexly represent contents, i.e. how to preprocess them and how to represent them
    In particular this class refers to *users*.

    Examples:

        >>> import clayrs.content_analyzer as ca
        >>> raw_source = ca.JSONFile(json_path)
        >>> users_config = ca.UserAnalyzerConfig(raw_source, id='user_id', output_directory='users_codified/')
        >>> # add single field config
        >>> users_config.add_single_config('occupation', FieldConfig(content_technique=ca.OriginalData()))
        >>> # add single exogenous technique
        >>> users_config.add_single_exogenous(ca.ExogenousConfig(ca.PropertiesFromDataset(field_name_list=['gender']))

    """
    def __str__(self):
        return str(self.__id)

    def __repr__(self):
        return f'UserAnalyzerConfig(source={self.__source}, ' \
               f'id={self.__id}, output directory={self.__output_directory}, ' \
               f'field_dict= {self.__field_dict}, exogenous representation={self.__exogenous_representation_list} ' \
               f'export_json={self.__export_json})'


class ItemAnalyzerConfig(ContentAnalyzerConfig):
    """
    Class that represents the configuration for the content analyzer. The configuration specifies how the
    `Content Analyzer` needs to complexly represent contents, i.e. how to preprocess them and how to represent them
    In particular this class refers to *items*.

    Examples:

        >>> import clayrs.content_analyzer as ca
        >>> raw_source = ca.JSONFile(json_path)
        >>> movies_config = ca.ItemAnalyzerConfig(raw_source, id='movie_id', output_directory='movies_codified/')
        >>> # add single field config
        >>> movies_config.add_single_config('occupation', FieldConfig(content_technique=ca.OriginalData()))
        >>> # add single exogenous technique
        >>> movies_config.add_single_exogenous(ca.ExogenousConfig(ca.PropertiesFromDataset(field_name_list=['gender']))
    """

    def __str__(self):
        return str(self.__id)

    def __repr__(self):
        return f'ItemAnalyzerConfig(source={self.__source}, ' \
               f'id={self.__id}, output directory={self.__output_directory}, ' \
               f'field_dict= {self.__field_dict}, exogenous representation={self.__exogenous_representation_list} ' \
               f'export_json={self.__export_json})'

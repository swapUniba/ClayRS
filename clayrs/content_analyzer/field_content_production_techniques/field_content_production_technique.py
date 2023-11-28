from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Union, Callable, Optional, TYPE_CHECKING, Tuple

import numpy as np
import requests
import validators
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar, get_iterator_thread

if TYPE_CHECKING:
    from clayrs.content_analyzer.content_representation.content import FieldRepresentation

from clayrs.content_analyzer.content_representation.content import FeaturesBagField, SimpleField, EmbeddingField
from clayrs.content_analyzer.information_processor.preprocessors.information_processor_abstract import InformationProcessor, \
    FileProcessor
from clayrs.content_analyzer.raw_information_source import RawInformationSource
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized


class FieldContentProductionTechnique(ABC):
    """
    Generic abstract class used to define the techniques that can be applied to the content's fields in order to
    produce their complex semantic representations.

    The FieldContentProductionTechnique creates, for each given content's raw data, the field's representation for a
    specific field
    """

    @staticmethod
    def process_data(data: str, preprocessor_list: List[InformationProcessor]) -> Union[List[str], str]:
        """
        The data passed as argument is processed using the preprocessor list (also given as argument) and is
        then returned

        Args:
            data (str): data on which each preprocessor, in the preprocessor list, will be used
            preprocessor_list (List[InformationProcessor]): list of preprocessors to apply to the data

        Returns:
            processed_data (Union[List[str], str): it could be a list of tokens created from the original data
            or the data in string form
        """
        processed_data = data
        for preprocessor in preprocessor_list:
            processed_data = preprocessor.process(processed_data)

        return processed_data

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        """
        Abstract method that defines the methodology used by a technique to produce a list of FieldRepresentation.
        This list of FieldRepresentation objects contains all the complex representations referring to a specific
        field for each content. So if there were 3 contents (in their original/raw form) passed to the method in total,
        there would be 3 elements in the list (one for each content) and each one of these elements would be the
        representation generated by the technique. The data contained in the field for each content is also
        pre-processed by a list of information processors

        EXAMPLE

            contents to process: content1, content2, content3.
            each content contains a 'Plot' and a 'Title' field.
            if the goal is to create the representations for the field 'Plot' the method produce_content will take

            produce_content('Plot', [], raw data of the contents)

            this method will produce a list in the following form:

            [FieldRepresentation for content1, FieldRepresentation for content2, FieldRepresentation for content3]
            where each FieldRepresentation refers to the 'Plot' field of the designated content

            in this example the processor list didn't contain any information processor. If a processor list containing
            information processors was defined, the method would be:

            produce_content('Plot', [NLTK()], raw data of the contents)

            the NLTK() processor will be applied to the data contained in the field 'Plot' for each content and, after
            this process, the field representation will be elaborated.

        Args:
             field_name (str): name of the contents' field on which the technique will be applied
             preprocessor_list (List[InformationProcessor]): list of information processors that will pre-process the
                data contained in the field for each content
            source (RawInformationSource): source where the raw data of the contents is stored

        Returns:
            representation_list(List[FieldRepresentation]): list containing the representations generated by the
                technique for each content on a specific field
        """
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class TextualContentTechnique(FieldContentProductionTechnique):
    """
    Class which encapsulates the logic of techniques which use text data as input
    The textual data is directly extracted from the raw source for the specified field_name
    """

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class FileContentTechnique(FieldContentProductionTechnique):
    """
    Class which encapsulates the logic for techniques which use data extracted from files as input

    In order for the framework to process files the input source (a CSV file, for example) should contain
    a field with all values of a certain type, that are either:

        - paths to files stored locally (either relative or absolute paths)
        - links to files

    In the case of file paths, the contents will simply be loaded. In the case of links, the files will first be
    downloaded locally and then loaded

    Args:

        contents_dirs: directory where the files are stored (or will be stored in the case of fields containing links)
        time_tuple: start and end second from which the data will be extracted
        max_timeout: maximum time to wait before considering a request failed (file from link)
        max_retries: maximum number of retries to retrieve a file from a link
        max_workers: maximum number of workers for parallelism
        batch_size: batch size for the dataloader
    """

    def __init__(self, contents_dirs: str = "contents_dirs", time_tuple: Tuple[Optional[int], Optional[int]] = (1, None), max_timeout: int = 2,
                 max_retries: int = 5, max_workers: int = 0, batch_size: int = 64, flatten: bool = False):

        self.contents_dirs = contents_dirs
        self.max_timeout = max_timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.flatten = flatten
        self.time_tuple = time_tuple

    def _retrieve_contents(self, field_name: str, raw_source: RawInformationSource):
        """
        Method which retrieves all the files for the specified field name in the raw source

        It checks all possible options, which are:

            - Local relative path to a file
            - Local absolute path to a file
            - Web link to a file
        """

        def dl_and_save_contents(url_or_path):

            filename = Path(url_or_path).name
            filename_no_extension = Path(url_or_path).stem

            content_path = os.path.join(field_contents_dir, filename)
            content_path_lnk = os.path.join(field_contents_dir, filename_no_extension + ".lnk")

            # if content_path or content_path_lnk already exist return them
            if os.path.isfile(content_path):
                return content_path
            elif os.path.isfile(content_path_lnk):
                return content_path_lnk

            if validators.url(url_or_path):

                n_retry = 0

                # keep trying to download a file until the max_retries number is reached
                while True:
                    try:
                        byte_content = requests.get(url_or_path, timeout=self.max_timeout).content
                        filename = Path(urlparse(url_or_path).path).name
                        content_path = os.path.join(self.contents_dirs, field_name, filename)
                        with open(content_path, 'wb') as f:
                            f.write(byte_content)

                        return content_path

                    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                        if n_retry < self.max_retries:
                            n_retry += 1
                        else:
                            logger.warning(f"Max number of retries reached ({n_retry + 1}/{self.max_retries}) for URL: "
                                           f"{url_or_path}\nThe content will be skipped")
                            return None
            else:

                # if the path is not absolute, we go in this if
                if not os.path.isfile(url_or_path):
                    # we build the absolute path and check again if the file exists
                    path_dir_contents = str(Path(raw_source.file_path).parent.absolute())
                    url_or_path = str(os.path.join(path_dir_contents, url_or_path))

                    if not os.path.isfile(url_or_path):
                        return None

                file_link = Path(url_or_path).stem + ".lnk"
                os.link(url_or_path, os.path.join(self.contents_dirs, field_name, file_link))

                return os.path.join(self.contents_dirs, field_name, file_link)

        field_contents_dir = os.path.join(self.contents_dirs, field_name)

        os.makedirs(field_contents_dir, exist_ok=True)

        url_contents = (content[field_name] for content in raw_source)

        content_paths = []
        error_count = 0
        with get_iterator_thread(self.max_workers, dl_and_save_contents, url_contents,
                                 keep_order=True, progress_bar=True, total=len(raw_source)) as pbar:

            pbar.set_description("Downloading/Locating contents")

            for future in pbar:
                if not future:
                    error_count += 1
                    content_paths.append(None)
                else:
                    content_paths.append(future)

        # when some files are not correctly loaded, the system loads a default value
        # depending on the file type (empty image, empty waveform, ...)
        # this warning message is sent to warn the user that this has happened
        if error_count != 0:
            logger.warning(f"Number of contents that couldn't be retrieved: {error_count}")

        return content_paths

    @abstractmethod
    def get_data_loader(self, field_name: str, raw_source: RawInformationSource, preprocessor_list: List[FileProcessor]) -> Tuple[DataLoader, bool]:
        raise NotImplementedError

    @abstractmethod
    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class SingleContentTechnique(TextualContentTechnique):
    """
    Technique specialized in the production of representations that don't need any external information in order
    to be processed. This type of technique only considers the raw data within the content's field to create
    the complex representation
    """

    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        """
        This method creates a list of FieldRepresentation objects, where each object is associated to a content
        and a specific field from the ones in said content. In order to do so, a simple preprocessing operation
        is done on the original data of the field (for each content) followed by the creation of the complex
        representation using the processed data. The complex representations are stored in a list and returned.
        """
        representation_list: List[FieldRepresentation] = []

        with get_progbar(list(source)) as pbar:
            # it iterates over all contents contained in the source in order to retrieve the raw data
            # the data contained in the field_name is processed using each information processor in the processor_list
            # the data is passed to the method that will create the single representation
            for content_data in pbar:
                processed_data = self.process_data(content_data[field_name], preprocessor_list)
                representation_list.append(self.produce_single_repr(processed_data))

        return representation_list

    @abstractmethod
    def produce_single_repr(self, field_data: Union[List[str], str]) -> FieldRepresentation:
        """
        This method creates a single FieldRepresentation using the field data only (in the complete process made by the
        produce_content method, the data also goes through pre-processing)

        Args:
             field_data (str): data contained in a specific field

        Returns:
            FieldRepresentation: complex representation created using the field data
        """
        raise NotImplementedError


class CollectionBasedTechnique(TextualContentTechnique):
    """
    Technique specialized in the production of representations that are in need of the entire collection in order
    to be processed. This type of technique performs a refactoring operation on the original dataset,
    so that each content in the collection is modified accordingly to the technique's needs
    """

    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:
        """
        This method creates a list of FieldRepresentation objects, where each object is associated to a content
        and a specific field from the ones in said content. In order to do so, the entire dataset where the original
        contents are stored has to be modified (and also processed using the preproccesors). After that, it re-creates
        the content_id for each content and uses it to retrieve the content from the refactored dataset. Finally, the
        refactored dataset is deleted.
        """
        # the contents in the collection are modified by the technique
        # in this phase the data is also processed using the preprocessor_list
        dataset_len = self.dataset_refactor(source, field_name, preprocessor_list)

        representation_list: List[FieldRepresentation] = []

        # produces the representation, retrieving it from the dataset given the content's position in the refactored
        # dataset

        for i in range(0, dataset_len):
            representation_list.append(self.produce_single_repr(i))

        # once the operation is complete the refactored collection is deleted
        self.delete_refactored()

        return representation_list

    @abstractmethod
    def produce_single_repr(self, content_position: int) -> FieldRepresentation:
        """
        This method creates a single FieldRepresentation using the content position only. The content position is used
        to retrieve the corresponding content from the dataset modified during the content creation process.

        Args:
             content_position (int): position of the content in the dataset

        Returns:
            FieldRepresentation: complex representation created or retrieved from the refactored dataset using the
                content_position
        """
        raise NotImplementedError

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]) -> int:
        """
        This method restructures the raw data in a way functional to the final representation

        Args:
            information_source (RawInformationSource): source containing the raw data of the contents
            field_name (str): field interested in the content production process on which the refactor will be based on
            preprocessor_list (List[InformationProcessor]): list of preproccesors applied to the data in the field_name
                for each content during the refactoring operation

        Returns:
            length of the refactored dataset
        """
        raise NotImplementedError

    @abstractmethod
    def delete_refactored(self):
        """
        Once the content production phase is completed, the refactored dataset is useless. This method will delete the
        refactored dataset
        """
        raise NotImplementedError


class OriginalData(SingleContentTechnique):
    """
    Technique used to retrieve the original data within the content's raw source without applying any
    processing operation.

    Note that if specified, preprocessing operations will still be applied!

    This technique is particularly useful if the user wants to keep the original
    data of the contents

    Args:
        dtype: If specified, data will be cast to the chosen dtype

    """

    def __init__(self, dtype: Callable = str):
        super().__init__()
        self.__dtype = dtype

    def produce_single_repr(self, field_data: Union[List[str], str]) -> SimpleField:
        """
        The contents' raw data in the given field_name is extracted and stored in a SimpleField object.
        The SimpleField objects created are stored in a list which is then returned.
        No further operations are done on the data in order to keep it in the original form.
        Because of that the preprocessor_list is ignored and not used by this technique
        """
        return SimpleField(self.__dtype(check_not_tokenized(field_data)))

    def __str__(self):
        return "OriginalData"

    def __repr__(self):
        return f'OriginalData(dtype={self.__dtype})'


class FromNPY(FieldContentProductionTechnique):
    """
    Technique used to import a collection of numpy arrays where each row will be treated as a separate
    instance of data for the specified field

    In this case, the expected field data from the source is a string representing an integer (that is the row of the
    numpy collection corresponding to the representation associated to that content instance)

    Note that if specified, preprocessing operations will ***NOT*** be applied! Preprocessing is skipped with this
    technique

    This technique is particularly useful if the user wants to import data generated by a different library

    Args:
        npy_file_path: Path where the numpy collection is stored

    """

    def __init__(self, npy_file_path: str):

        self.npy_file_path = npy_file_path
        self.np_matrix = np.load(npy_file_path)

        if len(self.np_matrix) > 0:
            self.dim_if_missing = self.np_matrix[0].shape
        else:
            raise ValueError('Matrix should have at least 1 row')

        self._missing: Optional[int] = None

    def produce_content(self, field_name: str, preprocessor_list: List[InformationProcessor],
                        source: RawInformationSource) -> List[FieldRepresentation]:

        self._missing = 0

        representation_list: List[FieldRepresentation] = []

        with get_progbar(list(source)) as pbar:
            # it iterates over all contents contained in the source in order to retrieve the raw data
            # the data contained in the field_name is processed using each information processor in the processor_list
            # the data is passed to the method that will create the single representation
            for content_data in pbar:
                representation_list.append(self.produce_single_repr(content_data[field_name]))

        if self._missing > 0:
            logger.warning(f"{self._missing} items could not be mapped (non int index). Empty arrays will be used")

        return representation_list

    def produce_single_repr(self, field_data: Union[List[str], str]) -> EmbeddingField:
        try:
            index = int(field_data)
        except ValueError:
            raise ValueError(f'Values should be integers but {field_data} was found of type {type(field_data)}!')
        try:
            return EmbeddingField(self.np_matrix[index])
        except IndexError:
            raise IndexError(f'Specified index ({field_data}) is greater than number of '
                             f'rows in matrix ({self.np_matrix.shape[0]}') from None

    def __str__(self):
        return "FromNPY"

    def __repr__(self):
        return f'FromNPY(npy_file_path={self.npy_file_path})'


class TfIdfTechnique(CollectionBasedTechnique):
    """
    Abstract class that generalizes the implementations that produce a Bag of words with tf-idf metric
    """

    def __init__(self):
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._feature_names: Optional[List[str]] = None

    def produce_single_repr(self, content_position: int) -> FeaturesBagField:
        """
        Retrieves the tf-idf values, for terms in document in the defined content_position,
        from the pre-computed word - document matrix.
        """
        nonzero_feature_index = self._tfidf_matrix[content_position, :].nonzero()[1]

        tfidf_sparse = self._tfidf_matrix.getrow(content_position).tocsc()
        pos_word_tuple = [(pos, self._feature_names[pos]) for pos in nonzero_feature_index]

        return FeaturesBagField(tfidf_sparse, pos_word_tuple)

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]):
        raise NotImplementedError

    def delete_refactored(self):
        del self._tfidf_matrix
        del self._feature_names


class SynsetDocumentFrequency(CollectionBasedTechnique):
    """
    Abstract class that generalizes implementations that use synsets
    """

    def __init__(self):
        self._synset_matrix: Optional[csr_matrix] = None
        self._synset_names: Optional[List[str]] = None

    def produce_single_repr(self, content_position: int) -> FeaturesBagField:
        """
        Retrieves the tf-idf values, for terms in document in the defined content_position,
        from the pre-computed word - document matrix.
        """
        nonzero_feature_index = self._synset_matrix[content_position, :].nonzero()[1]

        count_dense = self._synset_matrix.getrow(content_position).tocsc()
        pos_word_tuple = [(pos, self._synset_names[pos]) for pos in nonzero_feature_index]

        return FeaturesBagField(count_dense, pos_word_tuple)

    @abstractmethod
    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]):
        raise NotImplementedError

    def delete_refactored(self):
        del self._synset_matrix
        del self._synset_names

    def __repr__(self):
        return f'SynsetDocumentFrequency()'

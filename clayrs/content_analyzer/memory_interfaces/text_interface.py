import os

from whoosh.analysis import SimpleAnalyzer
from whoosh.fields import Schema, TEXT, KEYWORD
from whoosh.index import create_in, open_dir
from whoosh.formats import Frequency
from whoosh.qparser import QueryParser, OrGroup, FieldsPlugin
from whoosh.query import Term, Or
from whoosh.scoring import TF_IDF, BM25F
from typing import Union, Dict
import math
import abc

from clayrs.content_analyzer.memory_interfaces.memory_interfaces import TextInterface


class IndexInterface(TextInterface):
    """
    Abstract class that takes care of serializing and deserializing text in an indexed structure
    using the Whoosh library

    Args:
        directory (str): Path of the directory where the content will be serialized
    """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.__doc = None  # document that is currently being created and will be added to the index
        self.__writer = None  # index writer
        self.__doc_index = 0  # current position the document will have in the index once it is serialized
        self.__schema_changed = False  # true if the schema has been changed, false otherwise

    @property
    @abc.abstractmethod
    def schema_type(self):
        """
        Whoosh uses a Schema that defines, for each field of the content, how to store the data. In the case of this
        project, every field will have the same structure and will share the same field type. This method returns
        said field type.
        """
        raise NotImplementedError

    def init_writing(self, delete_old: bool = False):
        """
        Creates the index locally (in the directory passed in the constructor) and initializes the index writer.
        If an index already exists in the directory, what happens depend on the attribute delete_old passed as argument

        Args:
            delete_old (bool): if True, the index that was in the same directory is destroyed and replaced;
                if False, the index is simply opened
        """
        if os.path.exists(self.directory):
            if delete_old:
                self.delete()
                os.mkdir(self.directory)
                ix = create_in(self.directory, Schema())
                self.__writer = ix.writer()
            else:
                ix = open_dir(self.directory)
                self.__writer = ix.writer()
                self.__doc_index = self.__writer.reader().doc_count()
        else:
            os.mkdir(self.directory)
            ix = create_in(self.directory, Schema())
            self.__writer = ix.writer()

    def new_content(self):
        """
        The new content is a document that will be indexed. In this case the document is a dictionary with
        the name of the field as key and the data inside the field as value
        """
        self.__doc = {}

    def new_field(self, field_name: str, field_data: object):
        """
        Adds a new field to the document that is being created. Since the index Schema is generated dynamically, if
        the field name is not in the Schema already it is added to it

        Args:
            field_name (str): Name of the new field
            field_data (object): Data to put into the field
        """
        if field_name not in open_dir(self.directory).schema.names():
            self.__writer.add_field(field_name, self.schema_type)
            self.__schema_changed = True
        self.__doc[field_name] = field_data

    def serialize_content(self) -> int:
        """
        Serializes the content in the index. If the schema changed, the writer will commit the changes to the schema
        before adding the document to the index. Once the document is indexed, it can be deleted from the IndexInterface
        and the document position in the index is returned
        """
        if self.__schema_changed:
            self.__writer.commit()
            self.__writer = open_dir(self.directory).writer()
            self.__schema_changed = False
        self.__writer.add_document(**self.__doc)
        del self.__doc
        self.__doc_index += 1
        return self.__doc_index - 1

    def stop_writing(self):
        """
        Stops the index writer and commits the operations
        """
        self.__writer.commit()
        del self.__writer

    def get_field(self, field_name: str, content_id: Union[str, int]) -> str:
        """
        Uses a search index to retrieve the content corresponding to the content_id (if it is a string) or in the
        corresponding position (if it is an integer), and returns the data in the field corresponding to the field_name

        Args:
            field_name (str): name of the field from which the data will be retrieved
            content_id (Union[str, int]): either the position or Id of the content that contains the specified field

        Returns:
            Data contained in the field of the content
        """
        ix = open_dir(self.directory)
        with ix.searcher() as searcher:
            if isinstance(content_id, str):
                query = Term("content_id", content_id)
                result = searcher.search(query)
                result = result[0][field_name]
            elif isinstance(content_id, int):
                result = searcher.reader().stored_fields(content_id)[field_name]
            return result

    def query(self, string_query: str, results_number: int, mask_list: list = None,
              candidate_list: list = None, classic_similarity: bool = True) -> dict:
        """
        Uses a search index to query the index in order to retrieve specific contents using a query expressed in string
        form

        Args:
            string_query: query expressed as a string
            results_number: number of results the searcher will return for the query
            mask_list: list of content_ids of items to ignore in the search process
            candidate_list: list of content_ids of items to consider in the search process,
                if it is not None only items in the list will be considered
            classic_similarity: if True, classic tf idf is used for scoring, otherwise BM25F is used

        Returns:
            results: the final results dictionary containing the results found from the search index for the
                query. The dictionary will be in the following form:

                    {content_id: {"item": item_dictionary, "score": item_score}, ...}

                content_id is the content_id for the corresponding item
                item_dictionary is the dictionary of the item containing the fields as keys and the contents as values.
                So it will be in the following form:

                    {"Plot": "this is the plot", "Genre": "this is the Genre"}

                The item_dictionary will not contain the content_id since it is already defined and used as key of the
                external dictionary
                items_score is the score given to the item for the query by the index searcher
        """
        ix = open_dir(self.directory)
        with ix.searcher(weighting=TF_IDF if classic_similarity else BM25F) as searcher:
            candidate_query_list = None
            mask_query_list = None

            # the mask list contains the content_id for the items to ignore in the searching process
            # from the mask list a mask query is created and it will be used by the searcher
            if mask_list is not None:
                mask_query_list = []
                for document in mask_list:
                    mask_query_list.append(Term("content_id", document))
                mask_query_list = Or(mask_query_list)

            # the candidate list contains the content_id for the items to consider in the searching process
            # from the candidate list a candidate query is created and it will be used by the searcher
            if candidate_list is not None:
                candidate_query_list = []
                for candidate in candidate_list:
                    candidate_query_list.append(Term("content_id", candidate))
                candidate_query_list = Or(candidate_query_list)

            schema = ix.schema
            parser = QueryParser("content_id", schema=schema, group=OrGroup)
            # regular expression to match the possible field styles
            # examples: "content_id" or "Genre#2" or "Genre#2#custom_id"
            parser.add_plugin(FieldsPlugin(r'(?P<text>[\w-]+(\#[\w-]+(\#[\w-]+)?)?|[*]):'))
            query = parser.parse(string_query)
            score_docs = \
                searcher.search(query, limit=results_number, filter=candidate_query_list, mask=mask_query_list)

            # creation of the results dictionary, This phase is necessary because the Hit objects returned by the
            # searcher as results need the reader inside the search index in order to return information
            # so it would be impossible to access a field or the score of the item from outside this method
            # because of that this dictionary containing the most important infos is created
            results = {}
            for hit in score_docs:
                hit_dict = dict(hit)
                content_id = hit_dict.pop("content_id")
                results[content_id] = {}
                results[content_id]["item"] = hit_dict
                results[content_id]["score"] = hit.score
            return results

    def get_tf_idf(self, field_name: str, content_id: Union[str, int]) -> Dict[str, float]:
        r"""
        Calculates the tf-idf for the words contained in the field of the content whose id
        is content_id (if it is a string) or in the given position (if it is an integer).

        The tf-idf computation formula is:

        $$
        tf \mbox{-} idf = (1 + log10(tf)) * log10(idf)
        $$

        Args:
            field_name: Name of the field containing the words for which calculate the tf-idf
            content_id: either the position or Id of the content that contains the specified field

        Returns:
            words_bag: Dictionary whose keys are the words contained in the field, and the
                corresponding values are the tf-idf values
        """
        ix = open_dir(self.directory)
        words_bag = {}
        with ix.searcher() as searcher:
            if isinstance(content_id, str):
                query = Term("content_id", content_id)
                doc_num = searcher.search(query).docnum(0)
            elif isinstance(content_id, int):
                doc_num = content_id

            # if the document has the field == "" (length == 0) then the bag of word is empty
            if len(searcher.ixreader.stored_fields(doc_num)[field_name]) > 0:
                # retrieves the frequency vector (used for tf)
                list_with_freq = [term_with_freq for term_with_freq
                                  in searcher.vector(doc_num, field_name).items_as("frequency")]
                for term, freq in list_with_freq:
                    tf = 1 + math.log10(freq)
                    idf = math.log10(searcher.doc_count()/searcher.doc_frequency(field_name, term))
                    words_bag[term] = tf*idf
        return words_bag

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError


class KeywordIndex(IndexInterface):
    """
    This class implements the schema_type method: KeyWord. This is useful for splitting the indexed text in a list of
    tokens. The Frequency vector is also added so that the tf calculation is possible. Commas is true in case of a
    "content_id" field data containing white spaces
    """

    def __init__(self, directory: str):
        super().__init__(directory)

    @property
    def schema_type(self):
        return KEYWORD(stored=True, commas=True, vector=Frequency())

    def __str__(self):
        return "KeywordIndex"

    def __repr__(self):
        return f'KeywordIndex(directory={self.directory})'


class SearchIndex(IndexInterface):
    """
    This class implements the schema_type method: Text. By using a SimpleAnalyzer for the field, the data is kept as
    much as the original as possible
    """

    def __init__(self, directory: str):
        super().__init__(directory)

    @property
    def schema_type(self):
        return TEXT(stored=True, analyzer=SimpleAnalyzer())

    def __str__(self):
        return "SearchIndex"

    def __repr__(self):
        return f'SearchIndex(directory={self.directory})'

import lucene
import math
import shutil

from java.nio.file import Paths
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.document import Document, Field, StringField, FieldType, TextField
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.index import DirectoryReader, Term

from orange_cb_recsys.content_analyzer.memory_interfaces.memory_interfaces import TextInterface


class IndexInterface(TextInterface):
    """
    Abstract class that takes care of serializing and deserializing text in an indexed structure
    This use lucene library

    Args:
        directory (str): Path of the directory where the content will be serialized
    """

    def __init__(self, directory: str):
        super().__init__(directory)
        self.__doc = None
        self.__writer = None
        self.__field_type_frequency = None
        self.__field_type_searching = None

    def __str__(self):
        return "IndexInterface"

    def init_writing(self):
        self.__field_type_searching = FieldType(TextField.TYPE_STORED)
        self.__field_type_frequency = FieldType(StringField.TYPE_STORED)
        self.__field_type_frequency.setStored(True)
        self.__field_type_frequency.setTokenized(False)
        self.__field_type_frequency.setStoreTermVectors(True)
        self.__field_type_frequency.setStoreTermVectorPositions(True)
        self.__field_type_frequency.\
            setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)
        fs_directory = SimpleFSDirectory(Paths.get(self.directory))
        self.__writer = IndexWriter(fs_directory, IndexWriterConfig())

    def new_content(self):
        """
        In the lucene index case the new content
        is a new document in the index
        """
        self.__doc = Document()

    def new_field(self, field_name: str, field_data):
        """
        Add a new field

        Args:
            field_name (str): Name of the new field
            field_data: Data to put into the field
        """
        if isinstance(field_data, list):
            for word in field_data:
                self.__doc.add(Field(field_name, word, self.__field_type_frequency))
        else:
            self.__doc.add(Field(field_name, field_data, self.__field_type_frequency))

    def new_searching_field(self, field_name, field_data):
        """
        Add a new searching field. It will be used by the search engine recommender

        Args:
            field_name (str): Name of the new field
            field_data: Data to put into the field
        """
        self.__doc.add(Field(field_name, field_data, self.__field_type_searching))

    def serialize_content(self):
        """
        Serialize the content
        """
        doc_index = self.__writer.addDocument(self.__doc)
        return doc_index - 1

    def stop_writing(self):
        """
        Stop the index writer and commit the operations
        """
        self.__writer.commit()
        self.__writer.close()

    def get_tf_idf(self, field_name: str, content_id: str):
        """
        Calculates the tf-idf for the words contained in the field of the content whose id
        is content_id

        Args:
            field_name (str): Name of the field containing the words for which calculate the tf-idf
            content_id (str): Id of the content that contains the specified field

        Returns:
             words_bag (Dict <str, float>):
             Dictionary whose keys are the words contained in the field,
             and the corresponding values are the tf-idf values.
        """
        searcher = IndexSearcher(
            DirectoryReader.open(SimpleFSDirectory(Paths.get(self.directory))))
        query = QueryParser(
            "testo_libero", KeywordAnalyzer()).parse("content_id:\"" + content_id + "\"")
        score_docs = searcher.search(query, 1).scoreDocs
        document_offset = -1
        for score_doc in score_docs:
            document_offset = score_doc.doc

        reader = searcher.getIndexReader()
        words_bag = {}
        term_vector = reader.getTermVector(document_offset, field_name)
        term_enum = term_vector.iterator()
        for term in BytesRefIterator.cast_(term_enum):
            term_text = term.utf8ToString()
            postings = term_enum.postings(None)
            postings.nextDoc()
            term_frequency = 1 + math.log10(postings.freq())  # normalized term frequency
            inverse_document_frequency = math.log10(reader.maxDoc() / reader.docFreq(Term(field_name, term)))
            tf_idf = term_frequency * inverse_document_frequency
            words_bag[term_text] = tf_idf

        reader.close()
        return words_bag

    def delete_index(self):
        shutil.rmtree(self.directory, ignore_errors=True)

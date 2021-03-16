import lzma
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Set
import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.utils.const import logger, progbar


class Category(Enum):
    """
    Enum class that represents the different categories that nodes in the graph can have

    If graphs need to be extended (add the possibility of a 'context' node for example),
    start here adding a new category.
    """
    From = 1
    To = 2
    Property = 3


class Graph(ABC):
    """
    Abstract class that generalize the concept of a Graph

    Every Graph "is born" from a rating dataframe
    """
    def __init__(self, source_frame: pd.DataFrame):
        self.create_graph()
        self.populate_from_dataframe(source_frame)

    @abstractmethod
    def populate_from_dataframe(self, source_frame: pd.DataFrame):
        """
        Populate the graph using a DataFrame if it is of the format requested by
        _check_columns()
        """
        raise NotImplementedError

    @staticmethod
    def _check_columns(df: pd.DataFrame):
        """
        Check if there are at least least 'from_id', 'to_id', 'score' columns in the DataFrame

        Args:
            df (pandas.DataFrame): DataFrame to check

        Returns:
            bool: False if there aren't 'from_id', 'to_id', 'score' columns, else True
        """
        if 'from_id' not in df.columns or 'to_id' not in df.columns or 'score' not in df.columns:
            return False
        return True

    @staticmethod
    def normalize_score(score: float) -> float:
        """
        Convert the score in the range [-1.0, 1.0] in a normalized weight [0.0, 1.0]
        Args:
            score (float): float in the range [-1.0, 1.0]

        Returns:
            float in the range [0.0, 1.0]
        """
        old_max = 1
        old_min = -1
        new_max = 1
        new_min = 0

        old_range = (old_max - old_min)
        new_range = (new_max - new_min)
        new_value = (((score - old_min) * new_range) / old_range) + new_min
        return new_value

    @abstractmethod
    def create_graph(self):
        """
        Instantiate the empty graph
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def from_nodes(self) -> Set[object]:
        """
        Returns a set of 'from' nodes
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def to_nodes(self) -> Set[object]:
        """
        Returns a set of 'to' nodes'
        """
        raise NotImplementedError

    @abstractmethod
    def add_from_node(self, node: object):
        """
        Add a 'from' node to the graph
        """
        raise NotImplementedError

    @abstractmethod
    def add_to_node(self, node: object):
        """
        Add a 'to' node to the graph
        """
        raise NotImplementedError

    @abstractmethod
    def link_from_to(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """
        Adds an edge between 'from_node' and 'to_node',
        if the nodes are not in the graph adds the nodes
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_data(self, start_node: object, final_node: object):
        """
        Get data of the link between two nodes
        It can be None if does not exist
        """
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[object]:
        """
        Get all predecessors of a node
        """
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[object]:
        """
        Get all successors of a node
        """
        raise NotImplementedError

    @abstractmethod
    def is_from_node(self, node: object) -> bool:
        """
        Returns True if node is a 'from' node, false otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def is_to_node(self, node: object) -> bool:
        """
        Returns True if node is a 'to' node, false otherwise
        """
        raise NotImplementedError

    def get_voted_contents(self, node: object) -> List[object]:
        """
        Get all voted contents of a specified node

        Given a node, all voted contents of said node are his successors that are
        'to' nodes or 'from' nodes (In case a user votes another user for example)

        Args:
            node (object): the node from which we want to extract the voted contents

        Returns:
            List of nodes representing the voted contents for the node passed as
            parameter
        """
        voted_contents = []
        for succ in self.get_successors(node):
            # We append only if the linked node is a 'to_node'
            # or a 'from_node' because if it is for example
            # a 'property_node' then it isn't a voted content but a property
            if self.is_from_node(succ) or self.is_to_node(succ):
                voted_contents.append(succ)
        return voted_contents

    @property
    @abstractmethod
    def _graph(self):
        """
        PRIVATE
        Return the beneath implementation of the graph.

        Useful when is necessary to calculate some metrics for the graph
        """
        raise NotImplementedError


class BipartiteGraph(Graph):
    """
    Abstract class that generalize the concept of a BipartiteGraph

    A BipartiteGraph is a Graph containing only 'from' and 'to' nodes are allowed.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
    """

    def __init__(self, source_frame: pd.DataFrame):
        super().__init__(source_frame)

    def populate_from_dataframe(self, source_frame: pd.DataFrame):
        """
        Populate the graph using a DataFrame.
        It must have a 'from_id', 'to_id' and 'score' column.

        We iterate every row, and create a weighted link for every user and item in the rating frame
        based on the score the user gave the item, creating the nodes if they don't exist.

        Args:
            source_frame (pd.DataFrame): the rating frame from where the graph will be populated
        """
        if self._check_columns(source_frame):
            for idx, row in progbar(source_frame.iterrows(),
                                    max_value=source_frame.__len__(),
                                    prefix="Populating Graph:"):
                self.link_from_to(row['from_id'], row['to_id'], self.normalize_score(row['score']))
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')


class TripartiteGraph(BipartiteGraph):
    """
    Abstract class that generalize the concept of a TripartiteGraph

    A TripartiteGraph is a Graph containing 'from', 'to' and 'property' nodes, but the latter ones
    are allowed only for 'to' nodes.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
        item_contents_dir (str): the path containing items serialized
        item_exo_representation (str): the exogenous representation we want to extract properties from
        item_exo_properties (list): the properties we want to extract from the exogenous representation
        default_score_label(str): the label of the link between 'from' and 'to' nodes.
                                Default is 'score_label'
        default_not_rated_value(float): the default value with which the link will be weighted
                                Default is 0.5
    """

    def __init__(self, source_frame: pd.DataFrame, item_contents_dir: str = None,
                 item_exo_representation: str = None, item_exo_properties: List[str] = None,
                 default_score_label: str = 'score_label', default_not_rated_value: float = 0.5):

        self.__default_score_label = default_score_label

        self.__not_rated_value = default_not_rated_value

        self.__item_exogenous_representation: str = item_exo_representation

        self.__item_exogenous_properties: List[str] = item_exo_properties

        self.__item_contents_dir: str = item_contents_dir

        super().__init__(source_frame)

    def populate_from_dataframe(self, source_frame: pd.DataFrame):
        """
        Populate the graph using a DataFrame.
        It must have a 'from_id', 'to_id' and 'score' column.

        We iterate every row, and create a weighted link for every user and item in the rating frame
        based on the score the user gave the item, creating the nodes if they don't exist.
        We also add properties to 'to' nodes if the item_contents_dir is specified

        Args:
            source_frame (pd.DataFrame): the rating frame from where the graph will be populated
        """
        if self._check_columns(source_frame):
            for idx, row in progbar(source_frame.iterrows(),
                                    max_value=source_frame.__len__(),
                                    prefix="Populating Graph:"):
                self.link_from_to(row['from_id'], row['to_id'], self.normalize_score(row['score']))
                if self.get_item_contents_dir() is not None:
                    self._add_item_properties(row)
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')

    def _add_item_properties(self, row: dict):
        """
        Private method that given a row containing the 'to_id' field, tries to load the content
        from the item_contents_dir and if succeeds, extract properties presents in the content loaded
        based on the 'item_exo_representation' and 'item_exo_properties' parameters passed in
        the constructor as such:

        'item_exo_representation' was passed, 'item_exo_properties' was passed:
        ---------> Extract from the representation passed, the properties passed
            EXAMPLE:
                item_exo_representation = 0
                item_exo_properties = ['producer', 'director']

                will extract the 'producer' and 'director' property from the representation '0'

        'item_exo_representation' was passed, 'item_exo_properties' was NOT passed:
        ---------> Extract from the representation passed, ALL properties present in said representation
            EXAMPLE:
                    item_exo_representation = 0

                    will extract ALL properties from the representation '0'

        'item_exo_representation' was NOT passed, 'item_exo_properties' was passed:
        ---------> Extract from ALL representations, the properties passed
            EXAMPLE:
                item_exo_properties = ['producer', 'director']

                will extract the 'producer' and 'director' property from ALL exogenous representations
                of the content

        Args:
            row (dict): dict-like parameter containing at least a 'to_id' field
        """
        filepath = os.path.join(self.__item_contents_dir, row['to_id'])
        content = self.load_content(filepath)
        if content is not None:
            # Provided representation and properties
            if self.get_item_exogenous_representation() is not None and \
                    self.get_item_exogenous_properties() is not None:
                self._prop_by_rep(content, row['to_id'],
                                  self.get_item_exogenous_representation(), self.get_item_exogenous_properties(),
                                  row)

            # Provided only the representation
            elif self.get_item_exogenous_representation() is not None and \
                    self.get_item_exogenous_properties() is None:
                self._all_prop_in_rep(content, row['to_id'],
                                      self.get_item_exogenous_representation(),
                                      row)

            # Provided only the properties
            elif self.get_item_exogenous_representation() is None and \
                    self.get_item_exogenous_properties() is not None:
                self._prop_in_all_rep(content, row['to_id'],
                                      self.get_item_exogenous_properties(),
                                      row)

    def _prop_by_rep(self, content: Content, node: object, exo_rep: str, exo_props: List[str], row: dict):
        """
        Private method that extracts from the 'content' loaded, the 'exo_props' passed
        from the 'exo_rep' passed, then creates a link between the 'node' passed and properties
        extracted.

            EXAMPLE:
                exo_rep = 0
                exo_props = ['producer', 'director']

                will extract the 'producer' and 'director' property from the representation '0'
                in the 'content' parameter and creates a link from the 'node' passed to said
                properties

        Args:
            content (Content): content loaded
            node (object): node to add properties to
            exo_rep (str): representation from where to extract the 'exo_props'
            exo_props (list): the properties list to extract from 'content'
            row (dict): dict-like object containing eventual score for the properties
        """
        properties = None
        try:
            properties = content.get_exogenous_rep(exo_rep).value
        except KeyError:
            logger.warning("Representation " + exo_rep + " not found for " + content.content_id)

        if properties is not None:
            for prop in exo_props:
                if prop in properties.keys():
                    preference = self.get_preference(prop, row)
                    self.link_prop_node(node, properties[prop], preference, prop)
                else:
                    logger.warning("Property " + prop + " not found for " + content.content_id)

    def _all_prop_in_rep(self, content, node, exo_rep, row):
        """
        Private method that extracts from the 'content' loaded, ALL properties
        from the 'exo_rep' passed, then creates a link between the 'node' passed and properties
        extracted.

            EXAMPLE:
                exo_rep = 0

                will extract ALL properties from the representation '0' in the 'content' parameter and
                creates a link from the 'node' passed to said properties

        Args:
            content (Content): content loaded
            node (object): node to add properties to
            exo_rep (str): representation from where to extract the 'exo_props'
            row (dict): dict-like object containing eventual score for the properties
        """
        properties = None

        try:
            properties = content.get_exogenous_rep(exo_rep).value
        except KeyError:
            logger.warning("Representation " + exo_rep + " not found for " + content.content_id)

        if properties is not None:
            for prop_key in properties.keys():
                preference = self.get_preference(prop_key, row)
                self.link_prop_node(node, properties[prop_key], preference, prop_key)

            if len(properties) == 0:
                logger.warning("The chosen representation doesn't have any property!")

    def _prop_in_all_rep(self, content, node, exo_props, row):
        """
        Private method that extracts from the 'content' loaded, the 'exo_props' passed from
        ALL exo representation of the content, then creates a link between the 'node' passed and properties
        extracted. To avoid conflicts with multiple representations containing same properties, the
        properties extracted will be renamed as name_prop + exo_rep:

            EXAMPLE:
                exo_props = ['producer', 'director']

                will extract 'producer' and 'director' properties from ALL exogenous representation in the 'content'
                parameter and creates a link from the 'node' passed to said properties.
                The properties will be renamed as 'producer_0', 'director_0', 'producer_1', 'director_1'
                if for example the content has those two properties in the 0 exogenous representation
                and 1 exogenous representation


        Args:
            content (Content): content loaded
            node (object): node to add properties to
            exo_props (list): the properties list to extract from 'content'
            row (dict): dict-like object containing eventual score for the properties
        """
        properties = None
        properties_not_found = []
        for rep in content.exogenous_rep_dict:
            for prop in exo_props:
                if prop in content.get_exogenous_rep(rep).value:
                    if properties is None:
                        properties = {}
                    # properties = {director_0: aaaaa, director_1:bbbbb}
                    properties[prop + "_" + rep] = content.get_exogenous_rep(rep).value[prop]
                else:
                    properties_not_found.append(prop)

        if properties is not None:
            for prop_key in properties.keys():
                # EX. producer_0 -> producer so I can search for preference
                # in the original frame source
                original_prop_name = '_'.join(prop_key.split('_')[:-1])
                preference = self.get_preference(original_prop_name, row)

                self.link_prop_node(node, properties[prop_key], preference, prop_key)

            if len(properties_not_found) != 0:
                for prop in properties_not_found:
                    logger.warning("Property " + prop + " not found for " + content.content_id)
        else:
            logger.warning("None of the property chosen was found for " + content.content_id)

    def get_item_exogenous_representation(self) -> str:
        """
        Getter for item_exogenous_representation
        """
        return self.__item_exogenous_representation

    def get_item_exogenous_properties(self) -> List[str]:
        """
        Getter for item_exogenous_properties
        """
        return self.__item_exogenous_properties

    def get_default_score_label(self) -> str:
        """
        Getter for default_score_label
        """
        return self.__default_score_label

    def get_item_contents_dir(self) -> str:
        """
        Getter for item_contents_dir
        """
        return self.__item_contents_dir

    def get_preference(self, label: str, preferences_dict: dict) -> float:
        """
        Get the score of the label in the preferences_dict.
        Returns the default 'not_rated_value' if the label is not present

        EXAMPLE:
            preferences_dict:
            {'from_id': 'u1', 'to_id': 'i1', 'score': 0.6, 'director': 'Nolan', 'director_score': 0.8}

            get_preference('director', preferences_dict) ---> 0.8

        Args:
            label(str): label we want to search in the preferences_dict
            preferences_dict (dict): dict-like parameter containing some information about 'from'
                                    and 'to' nodes.
        """
        ls = '{}_score'.format(label.lower())
        if ls in preferences_dict.keys():
            return preferences_dict[ls]
        return self.__not_rated_value

    @staticmethod
    def load_content(file_path: str) -> Content:
        """
        Load the serialized content from the file_path specified

        Args:
            file_path (str): path to the file to load
        """
        try:
            with lzma.open('{}.xz'.format(file_path), 'r') as file:
                content = pickle.load(file)
        except FileNotFoundError:
            content = None
        return content

    @property
    @abstractmethod
    def property_nodes(self) -> Set[object]:
        """
        Returns a set of 'property' nodes
        """
        raise NotImplementedError

    @abstractmethod
    def add_prop_node(self, node: object):
        """
        Add a 'property' node to the graph
        """
        raise NotImplementedError

    def add_to_tree(self, to_node: object):
        """
        Add a 'to' node if is not in the graph and load properties from disk
        if the node has some
        The method will try to load the content from the item_contents_dir and extract
        from the loaded content the properties specified in the constructor (item_exo_representation,
        item_exo_properties)

        Args:
            to_node (object): 'to' node to add to the graph with its properties
        """
        self.add_to_node(to_node)
        if self.get_item_contents_dir() is not None:
            self._add_item_properties({'to_id': to_node})
        else:
            logger.warning("The dir is not specified! The node will be added with no "
                           "properties")

    @abstractmethod
    def link_prop_node(self, to_node: object, prop: object, weight: float, label: str):
        """
        Adds an edge between 'to_node' and a 'prop_node',
        if the nodes are not in the graph adds the nodes
        """
        raise NotImplementedError

    def get_properties(self, node: object) -> List[object]:
        """
        Get all properties of a specified node

        Given a node, all properties of said node are his successors that are
        'property' nodes

        Args:
            node (object): the node from which we want to extract the properties

        Returns:
            List of nodes representing the properties for the node passed as
            parameter
        """
        properties = []
        for succ in self.get_successors(node):
            if self.is_property_node(succ):
                link_data = self.get_link_data(node, succ)
                prop: dict = {link_data['label']: succ}
                properties.append(prop)
        return properties

    @abstractmethod
    def is_property_node(self, node: object) -> bool:
        """
        Returns True if the node is a 'property' node, False otherwise
        """
        raise NotImplementedError


class FullGraph(TripartiteGraph):
    """
    Abstract class that generalize the concept of a FullGraph

    A FullGraph is a Graph containing 'from', 'to' and 'property' nodes,
    and properties can be added to both 'from' and 'to' nodes.

    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
        user_contents_dir (str): the path containing users serialized
        item_contents_dir (str): the path containing items serialized
        user_exo_representation (str): the exogenous representation we want to extract properties from for the users
        user_exo_properties (list): the properties we want to extract from the exogenous representation for the users
        item_exo_representation (str): the exogenous representation we want to extract properties from for the items
        item_exo_properties (list): the properties we want to extract from the exogenous representation for the items
        default_score_label(str): the label of the link between 'from' and 'to' nodes.
                                Default is 'score_label'
        default_not_rated_value(float): the default value with which the link will be weighted
                                Default is 0.5
    """

    def __init__(self, source_frame: pd.DataFrame, user_contents_dir: str = None, item_contents_dir: str = None,
                 user_exo_representation: str = None, user_exo_properties: List[str] = None,
                 item_exo_representation: str = None, item_exo_properties: List[str] = None,
                 default_score_label: str = 'score_label', default_not_rated_value: float = 0.5):

        self.__user_exogenous_representation: str = user_exo_representation

        self.__user_exogenous_properties: List[str] = user_exo_properties

        self.__user_contents_dir: str = user_contents_dir

        super().__init__(source_frame, item_contents_dir,
                         item_exo_representation, item_exo_properties,
                         default_score_label, default_not_rated_value)

    def populate_from_dataframe(self, source_frame: pd.DataFrame):
        """
        Populate the graph using a DataFrame.
        It must have a 'from_id', 'to_id' and 'score' column.

        We iterate every row, and create a weighted link for every user and item in the rating frame
        based on the score the user gave the item, creating the nodes if they don't exist.
        We also add properties to 'to' nodes if the item_contents_dir is specified,
        and add properties to 'from' nodes if the user_contents_dir is specified.

        Args:
            source_frame (pd.DataFrame): the rating frame from where the graph will be populated
        """
        if self._check_columns(source_frame):
            for idx, row in progbar(source_frame.iterrows(),
                                    max_value=source_frame.__len__(),
                                    prefix="Populating Graph:"):

                self.link_from_to(row['from_id'], row['to_id'], self.normalize_score(row['score']),
                                  label=self.get_default_score_label())
                if self.get_item_contents_dir() is not None:
                    self._add_item_properties(row)

                if self.get_user_contents_dir() is not None:
                    self._add_usr_properties(row)
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')

    def _add_usr_properties(self, row):
        """
        Private method that given a row containing the 'from_id' field, tries to load the content
        from the user_contents_dir and if succeeds, extract properties presents in the content loaded
        based on the 'user_exo_representation' and 'user_exo_properties' parameters passed in
        the constructor as such:

        'user_exo_representation' was passed, 'user_exo_properties' was passed:
        ---------> Extract from the representation passed, the properties passed
            EXAMPLE:
                user_exo_representation = 0
                user_exo_properties = ['gender', 'birthdate']

                will extract the 'gender' and 'birthdate' property from the representation '0'

        'user_exo_representation' was passed, 'user_exo_properties' was NOT passed:
        ---------> Extract from the representation passed, ALL properties present in said representation
            EXAMPLE:
                    user_exo_representation = 0

                    will extract ALL properties from the representation '0'

        'user_exo_representation' was NOT passed, 'user_exo_properties' was passed:
        ---------> Extract from ALL representations, the properties passed
            EXAMPLE:
                user_exo_properties = ['gender', 'birthdate']

                will extract the 'gender' and 'birthdate' property from ALL exogenous representations
                of the content

        Args:
            row (dict): dict-like parameter containing at least a 'from_id' field
        """
        filepath = os.path.join(self.__user_contents_dir, row['from_id'])
        content = self.load_content(filepath)

        if content is not None:
            # Provided representation and properties
            if self.get_user_exogenous_representation() is not None and \
                    self.get_user_exogenous_properties() is not None:
                self._prop_by_rep(content, row['from_id'],
                                  self.get_user_exogenous_representation(), self.get_user_exogenous_properties(),
                                  row)

            # Provided only the representation
            elif self.get_user_exogenous_representation() is not None and \
                    self.get_user_exogenous_properties() is None:
                self._all_prop_in_rep(content, row['from_id'],
                                      self.get_user_exogenous_representation(),
                                      row)

            # Provided only the properties
            elif self.get_user_exogenous_representation() is None and \
                    self.get_user_exogenous_properties() is not None:
                self._prop_in_all_rep(content, row['from_id'],
                                      self.get_user_exogenous_properties(),
                                      row)

    def get_user_exogenous_representation(self) -> str:
        """
        Getter for user_exogenous_representation
        """
        return self.__user_exogenous_representation

    def get_user_exogenous_properties(self) -> List[str]:
        """
        Getter for user_exogenous_properties
        """
        return self.__user_exogenous_properties

    def get_user_contents_dir(self) -> str:
        """
        Getter for user_contents_dir
        """
        return self.__user_contents_dir

    @abstractmethod
    def link_prop_node(self, node: object, prop: object, weight: float, label: str):
        """
        Adds an edge between a generic node and a 'prop_node'
        the generic node must be in the graph otherwise no link will be created
        """
        raise NotImplementedError

    def add_from_tree(self, from_node: object):
        """
        Add a 'from' node if is not in the graph and load properties from disk
        if the node has some
        The method will try to load the content from the user_contents_dir and extract
        from the loaded content the properties specified in the constructor (user_exo_representation,
        user_exo_properties)

        Args:
            from_node (object): 'from' node to add to the graph with its properties
        """
        self.add_from_node(from_node)
        if self.get_user_contents_dir() is not None:
            self._add_usr_properties({'from_id': from_node})
        else:
            logger.warning("The dir is not specified for the users! The node will be added with no "
                           "properties")
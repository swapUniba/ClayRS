import lzma
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content


class Graph(ABC):
    """
    Abstract class that generalize the concept of a Graph
    """
    def __init__(self, source_frame: pd.DataFrame):
        self.__source_frame: pd.DataFrame = source_frame
        self.__graph = None

    @property
    def graph(self):
        return self.__graph

    @staticmethod
    def check_columns(df: pd.DataFrame):
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
        return 1 - (score + 1) / 2

    @property
    def source_frame(self):
        return self.__source_frame

    def get_from_nodes(self) -> List[str]:
        return list(self.__source_frame.from_id)

    def is_from_node(self, node) -> bool:
        return node in self.get_from_nodes()

    def get_to_nodes(self) -> List[str]:
        return list(self.__source_frame.to_id)

    def is_to_node(self, node) -> bool:
        return node in self.get_to_nodes()

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge_data(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError


class BipartiteGraph(Graph):
    """
    Abstract class that generalize the concept of a BipartiteGraph
    Attributes:
        source_frame (pandas.DataFrame): must contains at least 'from_id', 'to_id', 'score' columns. The graph will be
            generated from this DataFrame
    """
    def __init__(self, source_frame: pd.DataFrame):
        super().__init__(source_frame)
        self.__graph = None
        if self.check_columns(source_frame):
            self.create_graph()
            for idx, row in source_frame.iterrows():
                self.add_edge(row['from_id'], row['to_id'], self.normalize_score(row['score']))
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge_data(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError


class FullGraph(Graph):
    """ rating su più fields -> più archi (import di RatingsProcessor)"""
    def __init__(self, source_frame: pd.DataFrame, user_contents_dir: str = None, item_contents_dir: str = None,
                 user_exogenous_properties: List[str] = None, item_exogenous_properties: List[str] = None,
                 **options):

        self.__default_score_label = 'score_label'
        if 'default_score_label' in options.keys():
            self.__default_score_label = self.normalize_score(options['default_score_label'])

        self.__not_rated_value = 0.5
        if 'not_rated_value' in options.keys():
            self.__not_rated_value = self.normalize_score(options['not_rated_value'])

        self.__user_exogenous_properties: List[str] = user_exogenous_properties
        if user_exogenous_properties is None:
            self.__user_exogenous_properties: List[str] = []

        self.__item_exogenous_properties: List[str] = item_exogenous_properties
        if item_exogenous_properties is None:
            self.__item_exogenous_properties: List[str] = []

        self.__item_contents_dir: str = item_contents_dir
        self.__user_contents_dir: str = user_contents_dir
        super().__init__(source_frame)
        self.__graph = None

        if self.check_columns(source_frame):
            self.create_graph()

            for idx, row in self.source_frame.iterrows():
                self.add_edge(row['from_id'], row['to_id'], self.normalize_score(row['score']),
                              label=self.__default_score_label)
                content = self.load_content(row['to_id'])
                if content is not None:
                    for prop_name in self.__item_exogenous_properties:
                        try:
                            properties: dict = content.get_exogenous_rep(prop_name)
                        except KeyError:
                            properties = None

                        if properties is not None:

                            for prop_key in properties.keys():
                                preference = self.get_preference(prop_key, row)
                                self.add_edge(row['to_id'], properties[prop_key], preference, prop_key)

                content = self.load_content(row['from_id'])
                if content is not None:
                    for prop_name in self.__user_exogenous_properties:
                        try:
                            properties: dict = content.get_exogenous_rep(prop_name)
                        except KeyError:
                            properties = None

                        if properties is not None:

                            for prop_key in properties.keys():
                                preference = self.get_preference(prop_key, row)
                                self.add_edge(row['from_id'], properties[prop_key], preference, prop_key)
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')

    def get_user_exogenous_properties(self) -> List[str]:
        return self.__user_exogenous_properties

    def get_item_exogenous_properties(self) -> List[str]:
        return self.__item_exogenous_properties

    def get_default_score_label(self):
        return self.__default_score_label

    def get_item_contents_dir(self) -> str:
        return self.__item_contents_dir

    def get_user_contents_dir(self) -> str:
        return self.__user_contents_dir

    def get_preference(self, label: str, preferences_dict) -> float:
        ls = '{}_score'.format(label.lower())
        if ls in preferences_dict.keys():
            return preferences_dict[ls]
        return self.__not_rated_value

    def is_exogenous_property(self, node) -> bool:
        return not self.is_from_node(node) and not self.is_to_node(node)

    def load_content(self, file_name: str) -> Content:
        try:
            with lzma.open('{}/{}.xz'.format(self.__item_contents_dir, file_name), 'r') as file:
                content = pickle.load(file)
        except FileNotFoundError:
            try:
                with lzma.open('{}/{}.xz'.format(self.__user_contents_dir, file_name), 'r') as file:
                    content = pickle.load(file)
            except FileNotFoundError:
                content = None
        return content

    def query_frame(self, key: str, column: str) -> List[Dict]:
        """returns list of rows"""
        results_ = []
        rows = self.source_frame.loc[self.source_frame[column] == key]
        for row in rows.iterrows():
            results_.append(row)
        return results_

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    def add_tree(self, node: object):
        """
        add a node if is not in the graph and append properties if the node has some
        Args:
            node:

        Returns:

        """
        rows = self.query_frame(str(node), 'to_id')
        for row in rows:
            content = self.load_content(str(node))
            if content is not None:
                for prop_name in self.__item_exogenous_properties:
                    properties: dict = content.get_exogenous_rep(prop_name)

                    if properties is not None:

                        for prop_key in properties.keys():
                            preference = self.get_preference(prop_key, row)
                            self.add_edge(row['to_id'], properties[prop_key], preference, prop_key)

        rows = self.query_frame(str(node), 'from_id')
        for row in rows:
            content = self.load_content(str(node))
            if content is not None:
                for prop_name in self.__item_exogenous_properties:
                    properties: dict = content.get_exogenous_rep(prop_name)

                    if properties is not None:

                        for prop_key in properties.keys():
                            preference = self.get_preference(prop_key, row)
                            self.add_edge(row['from_id'], properties[prop_key], preference, prop_key)

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge_data(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    def get_properties(self, node: object) -> Dict[object, object]:
        properties = {}
        for succ in self.get_successors(node):
            edge_data = self.get_edge_data(node, succ)
            if edge_data['label'] != self.get_default_score_label():
                properties[edge_data['label']] = edge_data['weight']
        return properties

    def get_voted_contents(self, node: object) -> Dict[object, object]:
        properties = {}
        for succ in self.get_successors(node):
            edge_data = self.get_edge_data(node, succ)
            if edge_data['label'] == self.get_default_score_label():
                properties[edge_data['label']] = edge_data['weight']
        return properties


class TripartiteGraph(Graph):
    """ rating su più fields -> più archi (import di RatingsProcessor)"""
    def __init__(self, source_frame: pd.DataFrame, contents_dir: str = None, exogenous_properties: List[str] = None,
                 **options):

        self.__default_score_label = 'score_label'
        if 'default_score_label' in options.keys():
            self.__default_score_label = self.normalize_score(options['default_score_label'])

        self.__not_rated_value = 0.5
        if 'not_rated_value' in options.keys():
            self.__not_rated_value = self.normalize_score(options['not_rated_value'])

        self.__exogenous_properties: List[str] = exogenous_properties
        if exogenous_properties is None:
            self.__exogenous_properties: List[str] = []

        self.__contents_dir: str = contents_dir
        super().__init__(source_frame)
        self.__graph = None

        if self.check_columns(source_frame):
            self.create_graph()

            for idx, row in self.source_frame.iterrows():
                self.add_edge(row['from_id'], row['to_id'], self.normalize_score(row['score']),
                              label=self.__default_score_label)
                content = self.load_content(row['to_id'])
                if content is not None:
                    for prop_name in self.__exogenous_properties:
                        properties: dict = content.get_exogenous_rep(prop_name)

                        if properties is not None:

                            for prop_key in properties.keys():
                                preference = self.get_preference(prop_key, row)
                                self.add_edge(row['to_id'], properties[prop_key], preference, prop_key)
        else:
            raise ValueError('The source frame must contains at least \'from_id\', \'to_id\', \'score\' columns')

    def get_exogenous_properties(self) -> List[str]:
        return self.__exogenous_properties

    def get_default_score_label(self):
        return self.__default_score_label

    def get_contents_dir(self) -> str:
        return self.__contents_dir

    def get_preference(self, label: str, preferences_dict) -> float:
        ls = '{}_score'.format(label.lower())
        if ls in preferences_dict.keys():
            return preferences_dict[ls]
        return self.__not_rated_value

    def is_exogenous_property(self, node) -> bool:
        return not self.is_from_node(node) and not self.is_to_node(node)

    @staticmethod
    def load_content(file_name: str) -> Content:
        try:
            with lzma.open('{}.xz'.format(file_name), 'r') as file:
                content = pickle.load(file)
        except FileNotFoundError:
            content = None
        return content

    def query_frame(self, key: str, column: str) -> List[Dict]:
        """returns list of rows"""
        results_ = []
        rows = self.source_frame.loc[self.source_frame[column] == key]
        for row in rows.iterrows():
            results_.append(row)
        return results_

    @abstractmethod
    def create_graph(self):
        raise NotImplementedError

    @abstractmethod
    def add_node(self, node: object):
        raise NotImplementedError

    def add_tree(self, node: object):
        """
        add a node if is not in the graph and append properties if the node has some
        Args:
            node:

        Returns:

        """
        rows = self.query_frame(str(node), 'to_id')
        for row in rows:
            content = self.load_content(str(node))
            if content is not None:
                for prop_name in self.__exogenous_properties:
                    properties: dict = content.get_exogenous_rep(prop_name)

                    if properties is not None:

                        for prop_key in properties.keys():
                            preference = self.get_preference(prop_key, row)
                            self.add_edge(row['to_id'], properties[prop_key], preference, prop_key)

        rows = self.query_frame(str(node), 'from_id')
        for row in rows:
            content = self.load_content(str(node))
            if content is not None:
                for prop_name in self.__exogenous_properties:
                    properties: dict = content.get_exogenous_rep(prop_name)

                    if properties is not None:

                        for prop_key in properties.keys():
                            preference = self.get_preference(prop_key, row)
                            self.add_edge(row['from_id'], properties[prop_key], preference, prop_key)

    @abstractmethod
    def add_edge(self, from_node: object, to_node: object, weight: float, label: str = 'weight'):
        """ adds an edge, if the nodes are not in the graph, adds the nodes"""
        raise NotImplementedError

    @abstractmethod
    def get_edge_data(self, from_node: object, to_node: object):
        """it can be None if does not exist"""
        raise NotImplementedError

    @abstractmethod
    def get_adj(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_predecessors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_successors(self, node: object) -> List[Tuple[object, object, float]]:
        raise NotImplementedError

    def get_properties(self, node: object) -> Dict[object, object]:
        properties = {}
        for succ in self.get_successors(node):
            edge_data = self.get_edge_data(node, succ)
            if edge_data['label'] != self.get_default_score_label():
                properties[edge_data['label']] = edge_data['weight']
        return properties

    def get_voted_contents(self, node: object) -> Dict[object, object]:
        properties = {}
        for succ in self.get_successors(node):
            edge_data = self.get_edge_data(node, succ)
            if edge_data['label'] == self.get_default_score_label():
                properties[edge_data['label']] = edge_data['weight']
        return properties



from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
import numpy as np
from typing import List, TYPE_CHECKING
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from SPARQLWrapper.SPARQLExceptions import URITooLong
from babelpy.babelfy import BabelfyClient

if TYPE_CHECKING:
    from clayrs.content_analyzer.raw_information_source import RawInformationSource
    from clayrs.content_analyzer.content_representation.content import ExogenousPropertiesRepresentation

from clayrs.content_analyzer.content_representation.content import PropertiesDict, EntitiesProp
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_progbar
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized


class ExogenousPropertiesRetrieval(ABC):

    def __init__(self, mode: str = 'only_retrieved_evaluated'):
        """
        Class that creates a list of couples like this:
            <property: property value URI>
        The couples are properties retrieved from Linked Open Data Cloud

        Args:
            mode: one in: 'all', 'all_retrieved', 'only_retrieved_evaluated', 'original_retrieved',
        """
        self._check_mode(mode)
        self.__mode = mode

    @staticmethod
    def _check_mode(mode):
        modalities = {
            'all',
            'all_retrieved',
            'only_retrieved_evaluated',
            'original_retrieved',
        }
        if mode not in modalities:
            raise ValueError(f"mode={mode} not supported! Valid modalities are {modalities}")

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        self._check_mode(mode)
        self.__mode = mode

    @abstractmethod
    def get_properties(self, raw_source: RawInformationSource) -> List[ExogenousPropertiesRepresentation]:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class PropertiesFromDataset(ExogenousPropertiesRetrieval):
    """
    Exogenous technique which expands each content by using as external source the raw source itself

    Different modalities are available:

    * If `mode='only_retrieved_evaluated'` all fields for the content will be retrieved from raw source but discarding
    the ones with a blank value (i.e. '')

    ```title='JSON raw source'
    [{'Title': 'Jumanji', 'Year': 1995},
    {'Title': 'Toy Story', 'Year': ''}]
    ```

    ```python
    json_file = JSONFile(json_path)
    PropertiesFromDataset(mode='only_retrieved_evaluated').get_properties(json_file)
    # output is a list of PropertiesDict object with the following values:
    # [{'Title': 'Jumanji', 'Year': 1995},
    #  {'Title': 'Toy Story'}]
    ```

    * If `mode='all'` all fields for the content will be retrieved from raw source including the ones with a blank value

    ```title='JSON raw source'
    [{'Title': 'Jumanji', 'Year': 1995},
    {'Title': 'Toy Story', 'Year': ''}]
    ```

    ```python
    json_file = JSONFile(json_path)
    PropertiesFromDataset(mode='only_retrieved_evaluated').get_properties(json_file)
    # output is a list of PropertiesDict object with the following values:
    # [{'Title': 'Jumanji', 'Year': 1995},
    #  {'Title': 'Toy Story', 'Year': ''}]
    ```

    You could also choose exactly **which** fields to use to expand each content with the `field_name_list` parameter

    ```title='JSON raw source'
    [{'Title': 'Jumanji', 'Year': 1995},
    {'Title': 'Toy Story', 'Year': ''}]
    ```

    ```python
    json_file = JSONFile(json_path)
    PropertiesFromDataset(mode='only_retrieved_evaluated',
                          field_name_list=['Title']).get_properties(json_file)
    # output is a list of PropertiesDict object with the following values:
    # [{'Title': 'Jumanji'},
    #  {'Title': 'Toy Story'}]
    ```

    Args:

        mode: Parameter which specifies which properties should be retrieved.

            Possible values are ['only_retrieved_evaluated', 'all']:

                1. 'only retrieved evaluated' will retrieve properties which have a
                value, discarding ones with a blank value (i.e. '')
                2. 'all' will retrieve all properties, regardless if they have a value
                or not

        field_name_list: List of fields from the raw source that will be retrieved. Useful if you want to expand each
            content with only a subset of available properties from the local dataset

    """

    def __init__(self, mode: str = 'only_retrieved_evaluated', field_name_list: List[str] = None):
        super().__init__(mode)
        self.__field_name_list: List[str] = field_name_list

    def _check_mode(self, mode):
        modalities = {
            'all',
            'only_retrieved_evaluated',
        }
        if mode not in modalities:
            raise ValueError(f"mode={mode} not supported! Valid modalities are {modalities}")

    def get_properties(self, raw_source: RawInformationSource) -> List[PropertiesDict]:

        logger.info("Extracting exogenous properties from local dataset")
        prop_dict_list = []
        for raw_content in raw_source:

            if self.__field_name_list is None:
                prop_dict = raw_content
            else:
                prop_dict = {field: raw_content[field] for field in self.__field_name_list
                             if raw_content.get(field) is not None}

            if self.mode == 'only_retrieved_evaluated':
                prop_dict = {field: prop_dict[field] for field in prop_dict if prop_dict[field] != ''}

            prop_dict_list.append(PropertiesDict(prop_dict))

        return prop_dict_list

    def __str__(self):
        return "PropertiesFromDataset"

    def __repr__(self):
        return f'PropertiesFromDataset(mode={self.mode}, field_name_list={self.__field_name_list})'


class DBPediaMappingTechnique(ExogenousPropertiesRetrieval):
    """
    Exogenous technique which expands each content by using as external source the DBPedia ontology

    It needs the entity of the contents for which a mapping is required (e.g. entity_type=`dbo:Film`) and the field
    of the raw source that will be used for the actual mapping:


    Different modalities are available:

    * If `mode='only_retrieved_evaluated'`, all properties from DBPedia will be retrieved but discarding
    the ones with a blank value (i.e. '')

    * If `mode='all'`, all properties in DBPedia + all properties in local raw source will be retrieved.
    Local properties will be overwritten by dbpedia values if there's a conflict (same property in dbpedia and in local
    dataset)

    * If `mode='all_retrieved'`, all properties in DBPedia *only* will be retrieved

    * If `mode='original_retrieved'`, all local properties with their DBPedia value will be retrieved

    Args:
        entity_type: Domain of the contents you want to process (e.g. 'dbo:Film')
        label_field: Field of the raw source that will be used to map each content, DBPedia node with
            property **rdfs:label equal** to specified field value will be retrieved
        lang: Language of the `rdfs:label` that should match with `label_field` in the raw source
        mode: Parameter which specifies which properties should be retrieved.

            Possible values are ['only_retrieved_evaluated', 'all', 'all_retrieved', 'original_retrieved']:

                1. 'only retrieved evaluated' will retrieve properties which have a
                value, discarding ones with a blank value (i.e. '')
                2. 'all' will retrieve all properties from DBPedia + local source,
                regardless if they have a value or not
                3. 'all_retrieved' will retrieve all properties from DBPedia only
                4. 'original_retrieved' will retrieve all local properties with
                their DBPedia value
        return_prop_as_uri: If set to True, properties will be returned in their full uri form rather than in their
            rdfs:label form (e.g. "http://dbpedia.org/ontology/director" rather than "film director")
        max_timeout: Sometimes when mapping content to dbpedia, a batch of query may take longer than the max time
            allowed by the server due to internet issues: the framework will re-try the exact query `max_timeout` times
            before raising a `TimeoutError`
    """

    def __init__(self, entity_type: str, label_field: str, lang: str = 'EN',
                 mode: str = 'only_retrieved_evaluated', return_prop_as_uri: bool = False,
                 max_timeout: int = 5):
        super().__init__(mode)

        self._entity_type = entity_type
        self._label_field = label_field
        self._prop_as_uri = return_prop_as_uri
        self._lang = lang
        self._max_timeout = max_timeout

        self._sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        self._sparql.setReturnFormat(JSON)

        self._class_properties = self._get_properties_class()

    def _query_dbpedia(self, query: str):

        timeout_counter = 0
        self._sparql.setQuery(query)
        self._sparql.setMethod(GET)

        while True:
            try:
                result = self._sparql.query()
                if result.response.status == 200:
                    timeout_counter = 0
                    break
                elif result.response.status == 206:
                    if timeout_counter >= self._max_timeout:
                        raise TimeoutError("Maximum number of trials reached!")

                    logger.warning(f"Timeout occurred! - {timeout_counter + 1} out of {self._max_timeout} possible \n"
                                   f"The timeout counter will be reset as soon as the request goes through")
                    timeout_counter += 1
                    query += '\n'  # add space to avoid query plan cache so that we perform a new request
                    self._sparql.setQuery(query)
            except URITooLong:
                self._sparql.setMethod(POST)

        return result.convert()["results"]["bindings"]

    def _get_uris_all_contents(self, raw_source: RawInformationSource):
        prefixes = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "
        prefixes += "PREFIX dbo: <http://dbpedia.org/ontology/> "
        prefixes += "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        prefixes += "PREFIX foaf: <http://xmlns.com/foaf/0.1/> "

        all_contents_labels_original = [str(raw_content[self._label_field]) for raw_content in raw_source]

        select_clause = 'SELECT DISTINCT ?uri ?title '
        where_clause = 'WHERE { ' \
                       '?uri rdf:type dbo:Film . ' \
                       '?uri rdfs:label ?title . ' \
                       'BIND(str(?title) as ?str_label) . ' \
                       f'filter langMatches(lang(?title), "{self._lang}") '

        contents_taken = []
        results = []
        while True:
            contents_missing = set(all_contents_labels_original).difference(set(contents_taken))
            values_incomplete = ', '.join(f'"{wrapped}"' for wrapped in contents_missing)
            filter_query = f'filter(?str_label in ({values_incomplete}))'
            query_incomplete = prefixes + select_clause + where_clause + filter_query + "}"

            result_incomplete = self._query_dbpedia(query_incomplete)

            # if result_incomplete is empty, then there's no other item to map and we exit the loop
            if not result_incomplete:
                break

            results.extend(result_incomplete)
            contents_taken.extend([row["title"]["value"] for row in result_incomplete])

            logger.info(f"Contents mapped so far: {len(results)} of {len(all_contents_labels_original)} "
                        f"(ideal, tipically less are mapped)")

        logger.info(f"Contents mapped in total: {len(results)}")

        if len(results) == 0:
            raise ValueError("No mapping found")

        # Reset the order of contents
        res = {row["title"]["value"]: row["uri"]["value"] for row in results}
        uri_lables_dict = {'uri': [res[content] if res.get(content) else np.nan
                                   for content in all_contents_labels_original],
                           'label': all_contents_labels_original}

        results_df = pd.DataFrame.from_dict(uri_lables_dict)

        return results_df

    def _get_properties_class(self):
        query = "PREFIX dbo: <http://dbpedia.org/ontology/> "
        query += "SELECT DISTINCT ?property ?property_label WHERE { "
        query += "{ "
        query += "?property rdfs:domain ?class. "
        query += "%s rdfs:subClassOf+ ?class. " % self._entity_type
        query += "} UNION {"
        query += "?property rdfs:domain %s " % self._entity_type
        query += "} "
        query += "?property rdfs:label ?property_label. "
        query += f"FILTER (langMatches(lang(?property_label), \"{self._lang}\"))." + "} "

        results = self._query_dbpedia(query)

        if len(results) == 0:
            raise ValueError("The Entity type doesn't exists in DBPedia!")

        uri_labels_tuples = [(row["property"]["value"], row["property_label"]["value"])
                             for row in results]

        properties_df = pd.DataFrame.from_records(
            uri_labels_tuples,
            columns=['uri', 'label']
        )

        return properties_df

    def _retrieve_properties_contents(self, uris: pd.DataFrame):

        logger.info("Extracting properties for mapped contents...")

        query = "PREFIX dbo: <http://dbpedia.org/ontology/> "
        query += "SELECT ?uri ?property ?o WHERE {{ SELECT DISTINCT ?uri ?property ?o WHERE {"

        query += "VALUES ?property { "
        query += " ".join([f"<{uri_property}>" for uri_property in self._class_properties['uri']])
        query += "} "

        query += "VALUES ?uri { "
        query += " ".join([f"<{uri_item}>" for uri_item in uris['uri']])
        query += "} "

        query += "OPTIONAL {?uri ?property ?o . } "
        query += "} ORDER BY ?uri ?property ?o }} "

        results = []

        while True:
            # limit set at maximum allowed by dbpedia which is 10.000 may seem redundant
            # but the sparql endpoint won't work without it for this particular query
            query_incomplete = query + f" OFFSET {len(results)} LIMIT 10000"

            result_incomplete = self._query_dbpedia(query_incomplete)

            if not result_incomplete:
                break

            results.extend(result_incomplete)

        result_dict = defaultdict(lambda: defaultdict(list))
        for row in results:

            uri_item = row["uri"]["value"]
            property = row["property"]["value"]
            value = row.get("o", {}).get("value")  # safe way of accessing nested value in a dict

            # wrap every property value in a list, in case a property have more than one value
            # eg. starring: [DiCaprio, Tom Hardy]
            result_dict[uri_item][property].append(value)

        # index_map = {v: i for i, v in enumerate(list(uris['uri']))}
        # result_dict = dict(sorted(result_dict.items(), key=lambda pair: index_map[pair[0]]))

        # if some properties have only one value, then remove the list that wraps them
        # e.g. director: [Inarritu] -> director: Inarritu
        for uri in result_dict:
            result_dict[uri].update({prop: result_dict[uri][prop][0] for prop in result_dict[uri]
                                     if isinstance(result_dict[uri][prop], list) and len(result_dict[uri][prop]) == 1})

        return result_dict

    def _get_only_retrieved_evaluated(self, uris: pd.DataFrame, all_properties_dbpedia: dict) -> List[PropertiesDict]:

        prop_dict_list = []

        for uri in uris['uri']:

            if pd.notna(uri):

                content_properties_dbpedia = all_properties_dbpedia[uri]

                # Get only retrieved properties that have a value
                if self._prop_as_uri:
                    content_properties_final = {k: content_properties_dbpedia[k] for k in content_properties_dbpedia
                                                if content_properties_dbpedia[k] is not None}
                else:
                    # This goes into the class properties table and gets the labels to the corresponding uri
                    content_properties_final = {
                        self._class_properties.query('uri == @k')['label'].values[0]: content_properties_dbpedia[k]
                        for k in content_properties_dbpedia
                        if content_properties_dbpedia[k] is not None}

                prop_content = PropertiesDict(content_properties_final)
            else:
                prop_content = PropertiesDict({})

            prop_dict_list.append(prop_content)

        return prop_dict_list

    def _get_all_properties_retrieved(self, uris, all_properties_dbpedia) -> List[PropertiesDict]:
        prop_dict_list = []

        for uri in uris['uri']:

            if pd.notna(uri):

                content_properties_dbpedia = all_properties_dbpedia[uri]

                # Get all retrieved properties, so we substitute those with None with ""
                content_properties_final = {}
                for prop_uri in content_properties_dbpedia:

                    value = ''
                    if content_properties_dbpedia.get(prop_uri) is not None:
                        value = content_properties_dbpedia[prop_uri]

                    if self._prop_as_uri:
                        key = prop_uri
                    else:
                        # This goes into the class properties table and gets the labels to the corresponding uri
                        key = self._class_properties.query('uri == @prop_uri')['label'].values[0]

                    content_properties_final[key] = value

                prop_content = PropertiesDict(content_properties_final)
            else:
                prop_content = PropertiesDict({})

            prop_dict_list.append(prop_content)

        return prop_dict_list

    def __get_original_retrieved(self, uris, all_properties_dbpedia,
                                 raw_source: RawInformationSource) -> List[PropertiesDict]:

        prop_dict_list = []

        for uri, raw_content in zip(uris['uri'], raw_source):

            if pd.notna(uri):

                content_properties_dbpedia = all_properties_dbpedia[uri]
                content_properties_source = raw_content

                # Get all properties from source, those that have value in dbpedia will have value,
                # those that don't have value in dbpedia will be ''
                content_properties_final = {}
                for k in content_properties_source:
                    if k in self._class_properties['uri'].tolist():
                        value = content_properties_dbpedia[k] or ''

                        if self._prop_as_uri:
                            key = k
                        else:
                            key = self._class_properties.query('uri == @k')['label'].values[0]
                    elif k in self._class_properties['label'].tolist():
                        uri = self._class_properties.query('label == @k')['uri'].values[0]

                        value = content_properties_dbpedia[uri] or ''

                        if self._prop_as_uri:
                            key = uri
                        else:
                            key = k
                    else:
                        value = ''
                        key = k

                    content_properties_final[key] = value

                prop_content = PropertiesDict(content_properties_final)
            else:
                prop_content = PropertiesDict({})

            prop_dict_list.append(prop_content)

        return prop_dict_list

    def _get_all_properties(self, uris, all_properties_dbpedia,
                             raw_source: RawInformationSource) -> List[PropertiesDict]:
        prop_dict_list = []

        for uri, raw_content in zip(uris['uri'], raw_source):

            if pd.notna(uri):

                content_properties_dbpedia = all_properties_dbpedia[uri]
                content_properties_source = raw_content

                # Get all properties from source + all properties from dbpedia
                # if there are some properties in source that are also in dbpedia
                # the dbpedia value will overwrite the local source value
                content_properties_final = {}
                for k in content_properties_source:
                    if k in self._class_properties['uri'].tolist():
                        value = content_properties_dbpedia.pop(k)

                        if self._prop_as_uri:
                            key = k
                        else:
                            key = self._class_properties.query('uri == @k')['label'].values[0]

                    elif k in self._class_properties['label'].tolist():
                        uri = self._class_properties.query('label == @k')['uri'].values[0]

                        value = content_properties_dbpedia.pop(uri)

                        if self._prop_as_uri:
                            key = uri
                        else:
                            key = k
                    else:
                        value = content_properties_source[k]
                        key = k

                    if value is None:
                        value = content_properties_source[k]

                    content_properties_final[key] = value

                for k in content_properties_dbpedia:
                    value = content_properties_dbpedia[k]

                    if value is None:
                        value = ''

                    if self._prop_as_uri:
                        key = k
                    else:
                        key = self._class_properties.query('uri == @k')['label'].values[0]

                    content_properties_final[key] = value

                prop_content = PropertiesDict(content_properties_final)
            else:
                prop_content = PropertiesDict({})

            prop_dict_list.append(prop_content)

        return prop_dict_list

    def get_properties(self, raw_source: RawInformationSource) -> List[PropertiesDict]:
        logger.info("Extracting exogenous properties from DBPedia")
        uris = self._get_uris_all_contents(raw_source)

        uris_wo_none = uris.dropna()
        all_properties = self._retrieve_properties_contents(uris_wo_none)

        prop_dict_list = []
        if self.mode == 'only_retrieved_evaluated':
            prop_dict_list = self._get_only_retrieved_evaluated(uris, all_properties)

        elif self.mode == 'all_retrieved':
            prop_dict_list = self._get_all_properties_retrieved(uris, all_properties)

        elif self.mode == 'original_retrieved':
            prop_dict_list = self.__get_original_retrieved(uris, all_properties, raw_source)

        elif self.mode == 'all':
            prop_dict_list = self._get_all_properties(uris, all_properties, raw_source)

        return prop_dict_list

    def __str__(self):
        return "DBPediaMappingTechnique"

    def __repr__(self):
        return f'DBPediaMappingTechnique(mode={self.mode}, entity type={self._entity_type}, ' \
               f'label_field={self._label_field}, prop_as_uri={self._prop_as_uri}, max_timeout={self._max_timeout})'


class EntityLinking(ExogenousPropertiesRetrieval):
    """
    Abstract class that generalizes implementations that use entity linking for producing the semantic description
    """

    @abstractmethod
    def get_properties(self, raw_source: RawInformationSource) -> List[EntitiesProp]:
        raise NotImplementedError


class BabelPyEntityLinking(EntityLinking):
    """
    Exogenous technique which expands each content by using as external source the the BabelFy library.

    Each content will be expanded with the following babelfy properties (if available):

    * 'babelSynsetID',
    * 'DBPediaURL',
    * 'BabelNetURL',
    * 'score',
    * 'coherenceScore',
    * 'globalScore',
    * 'source'

    Args:
        field_to_link: Field of the raw source which will be used to search for the content properties in BabelFy
        api_key: String obtained by registering to babelfy website. If None only few queries can be executed
        lang: Language of the properties to retrieve
    """

    def __init__(self, field_to_link: str, api_key: str = None, lang: str = "EN"):
        super().__init__("all_retrieved")  # fixed mode since it doesn't make sense for babelfy
        self.__field_to_link = field_to_link
        self.__api_key = api_key
        self.__lang = lang
        self.__babel_client = BabelfyClient(self.__api_key, {"lang": lang})

    def get_properties(self, raw_source: RawInformationSource) -> List[EntitiesProp]:
        properties_list = []
        logger.info("Performing Entity Linking with BabelFy")
        with get_progbar(list(raw_source)) as pbar:
            for raw_content in pbar:
                data_to_disambiguate = check_not_tokenized(raw_content[self.__field_to_link])

                self.__babel_client.babelfy(data_to_disambiguate)

                properties_content = {}
                try:
                    if self.__babel_client.merged_entities is not None:

                        for entity in self.__babel_client.merged_entities:
                            properties_entity = {'babelSynsetID': '', 'DBPediaURL': '', 'BabelNetURL': '', 'score': '',
                                                 'coherenceScore': '', 'globalScore': '', 'source': ''}

                            for key in properties_entity:
                                if entity.get(key) is not None:
                                    properties_entity[key] = entity[key]

                            properties_content[entity['text']] = properties_entity

                    properties_list.append(EntitiesProp(properties_content))
                except AttributeError:
                    raise AttributeError("BabelFy limit reached! Insert an api key or change it if you inserted one!")

        return properties_list

    def __str__(self):
        return "BabelPyEntityLinking"

    def __repr__(self):
        return f'BabelPyEntityLinking(field_to_link={self.__field_to_link}, api_key={self.__api_key}, ' \
               f'lang={self.__lang})'

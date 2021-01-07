import os
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

from orange_cb_recsys.content_analyzer.content_representation.content import PropertiesDict
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.string_cleaner import clean_with_unders, clean_no_unders


class ExogenousPropertiesRetrieval(ABC):

    def __init__(self, mode: str = 'only_retrieved_evaluated'):
        """
        Class that creates a list of couples like this:
            <property: property value URI>
        The couples are properties retrieved from Linked Open Data Cloud

        Args:
            mode: one in: 'all', 'all_retrieved', 'only_retrieved_evaluated', 'original_retrieved',
        """
        self.__mode = self.__check_mode(mode)

    @staticmethod
    def __check_mode(mode):
        modalities = [
            'all',
            'all_retrieved',
            'only_retrieved_evaluated',
            'original_retrieved',
        ]
        if mode in modalities:
            return mode
        else:
            return 'all'

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        self.__mode = self.__check_mode(mode)

    @abstractmethod
    def get_properties(self, name, raw_content: Dict[str, object]) -> PropertiesDict:
        raise NotImplementedError


class PropertiesFromDataset(ExogenousPropertiesRetrieval):
    def __init__(self, mode: str = 'only_retrieved_evaluated', field_name_list: List[str] = None):
        super().__init__(mode)
        self.__field_name_list: List[str] = field_name_list

    def get_properties(self, name: str, raw_content: Dict[str, object]) -> PropertiesDict:

        logger.info("Extracting exogenous properties")
        prop_dict = {}
        for i, k in enumerate(raw_content.keys()):
            field_name = k
            if self.__field_name_list is not None:
                field_name = self.__field_name_list[i]
            prop_dict[field_name] = str(raw_content[k])
            if self.mode == 'only_retrieved_evaluated' and raw_content[k] != '':
                prop_dict.pop(k)
            elif self.mode == 'all_retrieved' or self.mode == 'all' or self.mode == 'original_retrieved':
                continue

        return PropertiesDict(name, prop_dict)


class DBPediaMappingTechnique(ExogenousPropertiesRetrieval):
    """
    Class that creates a list of couples like this:
        <property: property value URI>
    In this implementation the properties are retrieved from DBPedia

    Args:
        entity_type (str): domain of the items that you want to process
        lang (str): lang of the descriptions
        label_field: field ato be used as a filter,
            DBPedia node that has label value equal to specified field value
            will be retrieved
        additional_filters: other fields to use as filters,
            useful if label is not enough.
            You need to specify the name of the filed in your dataset
            and the name of the corresponding DBPedia property
        mode: one in: 'all', 'all_retrieved', 'only_retrieved_evaluated', 'original_retrieved',
    """

    def __init__(self, entity_type: str, lang: str, label_field: str, additional_filters=None,
                 mode: str = 'only_retrieved_evaluated'):
        super().__init__(mode)

        if additional_filters is None:
            additional_filters = {}

        self.__additional_filters = additional_filters
        self.__entity_type = entity_type
        self.__lang = lang
        self.__label_field = label_field

        self.__sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.__sparql.setReturnFormat(JSON)

        self.__has_label = self.__check_has_label()

    @property
    def label_field(self):
        return self.__label_field

    @label_field.setter
    def label_field(self, label_field: str):
        self.__label_field = label_field

    def __check_has_label(self):
        if len(self.__additional_filters) > 0:
            query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
            query += "SELECT DISTINCT "

            query += ', '.join("?%s" % field_name.lower() for field_name in self.__additional_filters)

            query += " WHERE { "

            query += '. '.join(
                ["?uri dbo:" + property_name + ' ?' + field_name.lower() + "_tmp" for field_name, property_name in
                 self.__additional_filters.items()]) + '. '

            query += ' '.join(
                ["OPTIONAL { ?" + field_name.lower() + "_tmp" + " rdfs:label" ' ?' + field_name.lower() + " }" for
                 field_name, property_name in
                 self.__additional_filters.items()])

            query += " } LIMIT 1 OFFSET 0"

            self.__sparql.setQuery(query)
            results = self.__sparql.query().convert()

            result = results["results"]["bindings"][0]

            return result.keys()
        else:
            return []

    def __mapping_query(self, raw_content):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT ?uri  "

        query += "WHERE { "

        # type matching
        query += "?uri rdf:type dbo:%s . " % self.__entity_type

        # label matching
        query += "?uri rdfs:label " + '?' + self.__label_field.lower()

        if (len(self.__additional_filters)) > 0:
            query += '. '

        # filter fields assignments
        query += '. '.join(["?uri dbo:%s ?%s. " % (property_name, field_name.lower()) +
                            "FILTER (" +
                            ' || '.join(["regex(?%s" % field_name.lower() +
                                         ("_label" if field_name.lower() in self.__has_label else '') +
                                         ', \"' + clean_no_unders(value) + '\", "i")'
                                         for value in (raw_content[field_name].split(', '))]) +
                            ")" for field_name, property_name in
                            self.__additional_filters.items()])

        if len(self.__has_label) != 0:
            query += '. '

        # label retrieval for fields with label
        query += '. '.join(
            ["?%s rdfs:label ?%s_label" % (field_name.lower(), field_name.lower())
             for field_name in self.__has_label])

        # lang filter
        query += ". FILTER langMatches(lang(?%s), \"%s\"). " % (self.__label_field.lower(), self.__lang)

        # label filter
        query += "FILTER regex(?%s, \"%s\", \"i\"). " % (
            self.__label_field.lower(), clean_no_unders(raw_content[self.__label_field]))

        query += " } "

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            raise ValueError("No mapping found")

        result = results["results"]["bindings"][0]
        uri = result["uri"]["value"]
        return uri

    def __get_properties_query(self):
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbo: <http://dbpedia.org/ontology/>  "
        query += "SELECT DISTINCT ?property_label WHERE { "
        query += "{ "
        query += "?property rdfs:domain ?class. "
        query += "dbo:%s rdfs:subClassOf+ ?class. " % self.__entity_type
        query += "} UNION {"
        query += "?property rdfs:domain dbo:%s" % self.__entity_type
        query += "} "
        query += "?property rdfs:label ?property_label. "
        query += "FILTER (langMatches(lang(?property_label), \"EN\")). }"

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        if len(results["results"]["bindings"]) == 0:
            return None
        property_labels = [clean_with_unders(row["property_label"]["value"])
                           for row in results["results"]["bindings"]]

        return property_labels

    def __retrieve_property_values(self, uri, new_property_labels):
        if uri is None:
            return None
        query = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        query += "SELECT ?p ?o WHERE { <%s> ?p_tmp ?o. ?p_tmp rdfs:label ?p }" % uri

        self.__sparql.setQuery(query)
        results = self.__sparql.query().convert()

        result_dict = {}
        for row in results["results"]["bindings"]:
            property_label = clean_with_unders(row["p"]["value"])

            if property_label in new_property_labels:
                result_dict[property_label] = row["o"]["value"]

        return result_dict

    def __get_only_retrieved_evaluated(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        new_property_labels = self.__get_properties_query()
        try:
            uri = self.__mapping_query(raw_content)
            result_dict = self.__retrieve_property_values(uri, new_property_labels)
        except ValueError:
            result_dict = {}
        return result_dict

    def __get_all_properties_retrieved(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        new_property_labels = self.__get_properties_query()
        result_dict = self.__get_only_retrieved_evaluated(raw_content)
        properties = {}
        for property_label in new_property_labels:
            if property_label in result_dict.keys():
                properties[property_label] = result_dict[property_label]
            else:
                properties[property_label] = ""
        return properties

    def __get_original_retrieved(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        original_property_labels = []
        original_properties = {}
        for key in raw_content.keys():
            original_property_labels.append(key)

        retrieved_properties = self.__get_only_retrieved_evaluated(raw_content)

        for property_label in original_property_labels:
            if property_label in retrieved_properties.keys():
                original_properties[property_label] = retrieved_properties[property_label]
            else:
                original_properties[property_label] = ""

        return original_properties

    def __get_all_properties(self, raw_content: Dict[str, object]) -> Dict[str, str]:
        all_prop_retrieved = self.__get_all_properties_retrieved(raw_content)
        property_labels = self.__get_properties_query()
        properties = {}
        for key in raw_content.keys():
            property_labels.append(key)

        for property_label in property_labels:
            if property_label in all_prop_retrieved.keys() and all_prop_retrieved[property_label] != '':
                properties[property_label] = all_prop_retrieved[property_label]
            elif property_label in raw_content.keys():
                properties[property_label] = raw_content[property_label]
            else:
                properties[property_label] = ""
        return properties

    def get_properties(self, name: str, raw_content: Dict[str, object]) -> PropertiesDict:
        """
        Execute the properties couple retrieval

        Args:
            name (str): string identifier of the returned properties object
            raw_content: represent a row in the dataset that
                is being processed

        Returns:
            PropertiesDict
        """
        logger.info("Extracting exogenous properties")
        prop_dict = {}
        if self.mode == 'only_retrieved_evaluated':
            prop_dict = self.__get_only_retrieved_evaluated(raw_content)

        if self.mode == 'all_retrieved':
            prop_dict = self.__get_all_properties_retrieved(raw_content)

        if self.mode == 'original_retrieved':
            prop_dict = self.__get_original_retrieved(raw_content)

        if self.mode == 'all':
            prop_dict = self.__get_all_properties(raw_content)

        print(prop_dict)

        return PropertiesDict(name, prop_dict)

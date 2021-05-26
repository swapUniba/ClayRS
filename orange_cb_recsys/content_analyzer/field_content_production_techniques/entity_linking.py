from babelpy.babelfy import BabelfyClient
from typing import List, Union

from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EntityLinking
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wraps some feature of Babelfy entity Linking.

    Args:
        api_key: string obtained by registering to babelfy website, with None babelpy key only few
            queries can be executed
    """

    def __init__(self, api_key: str = None):
        super().__init__()
        self.__api_key = api_key
        self.__babel_client = BabelfyClient(self.__api_key, {"lang": self.lang})

    def produce_single_repr(self, field_data: Union[List[str], str]) -> FeaturesBagField:
        """
        Produces a bag of features whose keys is babel net synset id and values are global score of the sysnset
        """
        field_data = check_not_tokenized(field_data)

        self.__babel_client.babelfy(field_data)
        feature_bag = {}
        try:
            if self.__babel_client.entities is not None:
                try:
                    for entity in self.__babel_client.entities:
                        feature_bag[entity['babelSynsetID']] = entity['globalScore']
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return FeaturesBagField(feature_bag)

    def __str__(self):
        return "BabelPyEntityLinking"

    def __repr__(self):
        return "< BabelPyEntityLinking: babel client = " + str(self.__babel_client) + " >"

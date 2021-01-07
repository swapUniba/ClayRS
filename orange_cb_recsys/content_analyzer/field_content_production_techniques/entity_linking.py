from babelpy.babelfy import BabelfyClient

from orange_cb_recsys.content_analyzer.content_representation.\
    content_field import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    EntityLinking, FieldContentProductionTechnique
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized


class BabelPyEntityLinking(EntityLinking):
    """
    Interface for the Babelpy library that wraps some feature of Babelfy entity Linking.

    Args:
        api_key: string obtained by registering to
            babelfy website, with None babelpy key only few
            queries can be executed
    """

    def __init__(self, api_key: str = None):
        super().__init__()
        self.__api_key = api_key
        self.__babel_client = None

    @FieldContentProductionTechnique.lang.setter
    def lang(self, lang: str):
        FieldContentProductionTechnique.lang.fset(self, lang)
        params = dict()
        params['lang'] = self.lang
        self.__babel_client = BabelfyClient(self.__api_key, params)

    def __str__(self):
        return "BabelPyEntityLinking"

    def produce_content(self, field_representation_name: str, field_data) -> FeaturesBagField:
        """
        Produces the field content for this representation,
        bag of features whose keys is babel net synset id and
        values are global score of the sysnset

        Args:
            field_representation_name (str): Name of the field representation
            field_data: Text that will be linked to BabelNet

        Returns:
            feature_bag (FeaturesBagField)
        """
        field_data = check_not_tokenized(field_data)

        self.__babel_client.babelfy(field_data)
        feature_bag = FeaturesBagField(field_representation_name)
        try:
            if self.__babel_client.entities is not None:
                try:
                    for entity in self.__babel_client.entities:
                        feature_bag.append_feature(entity['babelSynsetID'], entity['globalScore'])
                except AttributeError:
                    pass
        except AttributeError:
            pass


        return feature_bag

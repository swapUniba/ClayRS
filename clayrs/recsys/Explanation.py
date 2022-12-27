from abc import abstractmethod
from typing import List

from clayrs.content_analyzer import Rank, Ratings
from clayrs.recsys import NXFullGraph, ItemNode, PropertyNode, NXPageRank, TestRatingsMethodology


def explain(film_piaciuti, film_raccomandati):
    """
    1 memorizzo il profilo
    2 memorizzo i raccomandati
    3 creo il grafo
    4 calcolo il ranking delle proprietà
    4* seleziono il numero di proprietà da considerare
    5 creo la struttura con le triple usate per generare le spiegazioni (RDF)
    6 creo le spiegazioni
    """
    explanation = {}
    profile, numero_film1 = mapping_profilo(film_piaciuti)  # dizionario(titolo, uri) dei film contenuti nel profilo, |profile|
    recommendation, numero_film2 = mapping_profilo(film_raccomandati)  # dictionary (title, uri), len recommendation
    G, common_properties, numero_proprieta = costruisci_grafo(profile, recommendation) #graph, list common prop, num properties
    pagR=NXPageRank(alpha=0.85, personalized=False, max_iter=100, tol=1e-06, nstart=None, weight=True,
               relevance_threshold=None, rel_items_weight=0.8, rel_items_prop_weight=None, default_nodes_weight=0.2)
    pagR.rank({a}, G, test_set, recs_number=None, methodology=TestRatingsMethodology(), num_cpus=1)
    """for user_id in set(recs.user_id_column):

        #user_explanation = []

        # calcolare lista item piaciuti (item del profilo con rating > relevance_threshold)
        # se relevance_threshold None, calcolare voto medio dato dall'utente, gli item piaciuti
        # sono gli item con rating maggiore del voto medio

        user_relevant_items = []


        # i1 -> director-> nolan
        #    -> starring-> DiCaprio
        #    -> i2
        # ottengo il grafo con gli item e le proprietà

        #for relevant_item in user_relevant_items:
            #succ_item = self.graph.get_successors(ItemNode(relevant_item))
            #prop_list = [succ for succ in succ_item if isinstance(succ, PropertyNode)]

            # calcolare algoritmo explode

        # usare template di stringhe


        #all_explanation[user_id] = user_explanation"""




def mapping_profilo(id_film):
    profile_prov = {} #dizionario profilo provvisorio
    profile = {}
    id_key_dictionary, id_name_dictionary, numero_film = lettura_file("list_items_movies.mapping")
    for item in id_film:
        if item in id_key_dictionary.keys():
            profile_prov[item] = id_key_dictionary[item]
    for key, value in profile_prov.items():
        profile[value] = id_name_dictionary[value]


    return profile, numero_film


def lettura_file(file_path):
    id_key_dictionary = {} #dizionario (k id, val titolo)
    name_key_dictionary = {} #dizionari (k titolo, val uri film)
    numero_film = 0
    with open(file_path, 'r') as f:
        for line in f:
            numero_film += 1
            if line == '\n':
                break
            else:
                line = line.split('\t')
                id_key_dictionary[line[0]] = line[1]
                name_key_dictionary[line[1]] = line[2]

    return id_key_dictionary, name_key_dictionary, numero_film

def costruisci_grafo(profile, recommendation):
    profile_properties = {}
    recommendation_properties = {}
    numero_proprieta = 0

    with open('movies_stored_prop.mapping', 'r') as f:
        for line in f:                                    # scorro le righe del file con le triple RDF
            numero_proprieta += 1
            line = line.rstrip().split('\t')
            for key, value in profile.items():            # scorro i film piaciuti nel dizionario
                if line[0] == value:                      # quando trovo l'URI del film piaciuto in una tripla RDF
                    profile_properties[line[2]] = value             # inserisco la proprieta come chiave e il film come valore in un nuovo dizionario

    with open('movies_stored_prop.mapping', 'r') as f:
        for line in f:                                    # scorro le righe del file con le triple RDF
            line = line.rstrip().split('\t')
            for key, value in recommendation.items():     # scorro i film raccomandati nel dizionario
                if line[0] == value:                      # quando trovo l'URI del film piaciuto in una tripla RDF
                    recommendation_properties[line[2]] = value      # inserisco la proprieta come chiave e il film come valore in un nuovo dizionario di appoggio

    profile_common_prop = {}
    recomm_common_prop = {}

    for key, value in profile_properties.items():                   # scorro entrambi i dizionari appena creati
        if key in recommendation_properties.keys():                               # se film piaciuti e film raccomandati hanno proprieta in comune
            profile_common_prop[key] = value                     # inserisco proprieta e film in un dizionario (solo quelle in comune)
            recomm_common_prop[key] = recommendation_properties[key]

    common_properties = list(profile_common_prop.keys())           # creo una lista con solo le proprieta in comune
    G = NXFullGraph()                                     # creo un grafo orientato
    for key, value in profile_common_prop.items():
        G.add_edge(value, key)                            # aggiungo come nodi i film piaciuti e i film raccomandati
    for key, value in recomm_common_prop.items():            # aggiungo come nodi le proprieta in comune
        G.add_edge(key, value)                            # collego i nodi attraverso le proprieta in comune con archi

    return G, common_properties, numero_proprieta
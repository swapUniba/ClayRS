from abc import abstractmethod
from typing import List, Optional, Union, Set

from clayrs.content_analyzer import Rank, Ratings
from clayrs.recsys import NXFullGraph, ItemNode, PropertyNode, UserNode, NXTripartiteGraph

__all__ = ["Explod"]

from clayrs.utils.context_managers import get_progbar


class Explod:

    def __init__(self, items_directory: str, items_exo_properties: Union[set, dict]):
        self.items_dir = items_directory
        self.items_exo_props = items_exo_properties

        # set when fit
        self.graph: Optional[NXTripartiteGraph] = None

    def fit(self, original_ratings: Ratings, recommendations: Rank):

        # profile link will be the standard direction user -> item
        # e.g. u1 -> i1
        graph = NXTripartiteGraph(original_ratings,
                                  item_exo_properties=self.items_exo_props,
                                  item_contents_dir=self.items_dir,
                                  link_label="profile_link")

        graph.add_node_with_prop([ItemNode(rank_item_id) for rank_item_id in set(recommendations.item_id_column)],
                                 item_exo_properties=self.items_exo_props,
                                 item_contents_dir=self.items_dir)

        # recommendation link will be the inverse direction item -> user (needed to easily disambiguate)
        # i1 -> u1
        for interaction in recommendations:
            graph.add_link(ItemNode(interaction.item_id),
                           UserNode(interaction.user_id),
                           weight=interaction.score,
                           label="recs_link")

        self.graph = graph

        return self

    def compute_rank(self, prop_node: PropertyNode, user_profile: Set[ItemNode], user_recs: Set[ItemNode],
                     alpha: float = 0.5, beta: float = 0.5, idf: bool = True):
        num_link_profile = sum(1 for item_node in user_profile
                               if self.graph.get_link_data(item_node, prop_node) is not None)

        num_link_recs = sum(1 for item_node in user_recs
                            if self.graph.get_link_data(item_node, prop_node) is not None)

        score = None

        # only consider property nodes in common
        if num_link_profile != 0 and num_link_recs != 0:
            score = (alpha * (num_link_profile / len(user_profile))) + (beta * (num_link_recs / len(user_recs)))

            if idf:
                score *= 1  # idf to implement yet

        return score

    def explain(self, user_id: str, top_k: Optional[int] = 5, alpha: float = 0.5, beta: float = 0.5, idf: bool = True):

        # remember profile links are in the direction user -> item
        user_profile = [item_node for item_node in self.graph.get_successors(UserNode(user_id))
                        if isinstance(item_node, ItemNode)]

        # remember recommendation links are in the direction item <- user
        user_rank = [item_node for item_node in self.graph.get_predecessors(UserNode(user_id))
                     if isinstance(item_node, ItemNode)]

        unique_item_nodes = set(user_profile + user_rank)
        linked_prop_nodes = set(prop_node for item_node in unique_item_nodes
                                for prop_node in self.graph.get_successors(item_node)
                                if isinstance(prop_node, PropertyNode))

        with get_progbar(linked_prop_nodes, desc=f"Computing top-{top_k} properties for user {user_id}") as pbar:
            linked_prop_scores = {
                prop_node: self.compute_rank(prop_node, set(user_profile), set(user_rank), alpha, beta, idf)
                for prop_node in pbar

            }

        # remove properties which do not appear in both profile and rec (None value)
        linked_prop_scores = dict(filter(lambda item: item[1] is not None, linked_prop_scores.items()))

        # sort and return top-k properties
        linked_prop_scores = dict(sorted(linked_prop_scores.items(),
                                         key=lambda item: (item[1] is not None, item[1]), reverse=True)[:top_k])

        return linked_prop_scores


def explain(film_piaciuti, film_raccomandati):
    """
    explanation = {}
    for user_id in set(recs.user_id_column):

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

    """
       1 memorizzo il profilo
       2 memorizzo i raccomandati
       3 creo il grafo
       4 calcolo il ranking delle proprietà
       4* seleziono il numero di proprietà da considerare
       5 creo la struttura con le triple usate per generare le spiegazioni (RDF)
       6 creo le spiegazioni
       """
    profile, numero_film1 = mapping_profilo(
        film_piaciuti)  # dizionario(titolo, uri) dei film contenuti nel profilo, |profile|
    recommendation, numero_film2 = mapping_profilo(film_raccomandati)  # dictionary (title, uri), len recommendation
    G, common_properties, numero_proprieta = costruisci_grafo(profile,
                                                              recommendation)  # graph, list common prop, num properties

    """pagR=NXPageRank(alpha=0.85, personalized=False, max_iter=100, tol=1e-06, nstart=None, weight=True,
               relevance_threshold=None, rel_items_weight=0.8, rel_items_prop_weight=None, default_nodes_weight=0.2)
    test_set= ()
    pagR.rank({a}, G, test_set, recs_number=None, methodology=None, num_cpus=1)"""
    ranked_prop = ranking_proprieta(G, common_properties, profile, recommendation, idf=True)
    sorted_prop = proprieta_da_considerare(ranked_prop, 5)
    stampa_proprieta(sorted_prop)


def mapping_profilo(id_film):
    profile_prov = {}  # dizionario profilo provvisorio
    profile = {}
    id_key_dictionary, id_name_dictionary, numero_film = lettura_file("list_items_movies.mapping")
    for item in id_film:
        if item in id_key_dictionary.keys():
            profile_prov[item] = id_key_dictionary[item]
    for key, value in profile_prov.items():
        profile[value] = id_name_dictionary[value]

    return profile, numero_film


def lettura_file(file_path):
    id_key_dictionary = {}  # dizionario (k id, val titolo)
    name_key_dictionary = {}  # dizionari (k titolo, val uri film)
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
        for line in f:  # scorro le righe del file con le triple RDF
            numero_proprieta += 1
            line = line.rstrip().split('\t')
            for key, value in profile.items():  # scorro i film piaciuti nel dizionario
                if line[0] == value:  # quando trovo l'URI del film piaciuto in una tripla RDF
                    profile_properties[line[
                        2]] = value  # inserisco la proprieta come chiave e il film come valore in un nuovo dizionario

    with open('movies_stored_prop.mapping', 'r') as f:
        for line in f:  # scorro le righe del file con le triple RDF
            line = line.rstrip().split('\t')
            for key, value in recommendation.items():  # scorro i film raccomandati nel dizionario
                if line[0] == value:  # quando trovo l'URI del film piaciuto in una tripla RDF
                    recommendation_properties[line[
                        2]] = value  # inserisco la proprieta come chiave e il film come valore in un nuovo dizionario di appoggio

    profile_common_prop = {}
    recomm_common_prop = {}

    for key, value in profile_properties.items():  # scorro entrambi i dizionari appena creati
        if key in recommendation_properties.keys():  # se film piaciuti e film raccomandati hanno proprieta in comune
            profile_common_prop[key] = value  # inserisco proprieta e film in un dizionario (solo quelle in comune)
            recomm_common_prop[key] = recommendation_properties[key]

    common_properties = list(profile_common_prop.keys())  # creo una lista con solo le proprieta in comune
    G = NXFullGraph()  # creo un grafo orientato
    for key, value in profile_common_prop.items():
        G.add_link(value, key)  # aggiungo come nodi i film piaciuti e i film raccomandati
    for key, value in recomm_common_prop.items():  # aggiungo come nodi le proprieta in comune
        G.add_link(key, value)  # collego i nodi attraverso le proprieta in comune con archi

    return G, common_properties, numero_proprieta


# Funzione che prende in input il grafo costruito, le proprieta in comune e i due dizionari di film piaciuti e
# raccomandati  ed effettua il ranking delle proprieta in comune in ordine di
# influenza attraverso il calcolo di un punteggio (in ordine decrescente)
def ranking_proprieta(G, proprieta_comuni, item_piaciuti, item_raccom, idf):
    alfa = 0.5
    beta = 0.5
    score_prop = {}

    # if idf:
    for prop in proprieta_comuni:  # per ogni proprieta in comune, calcolo il numero di archi entranti
        if G.node_exists(prop):  # ed uscenti e li uso nella formula, insieme al rispettivo IDF
            num_in_edges = len(G.get_predecessors(prop))
            # G.in_degree(prop)
            # per calcolare il punteggio
            num_out_edges = len(G.get_successors(prop))
            # G.out_degree(prop)
            score_prop[prop] = ((alfa * num_in_edges / len(item_piaciuti)) + (beta * num_out_edges / len(item_raccom)))
            if idf:
                score_prop[prop] = score_prop[prop] * calcola_IDF(prop)

        sorted_prop = dict((sorted(score_prop.items(), key=lambda item: item[1],
                                   reverse=True)))  # ordino la lista di punteggi in ordine decrescente

    print("Le proprieta sono state rankate e ordinate con successo!\n")

    return sorted_prop


def proprieta_da_considerare(prop_rankate, numero_prop_considerate):
    prop_considerate = {}
    for prop, score in prop_rankate.items():
        prop_considerate[prop] = score
        if len(prop_considerate) == numero_prop_considerate:
            break
    return prop_considerate


def stampa_proprieta(proprieta):
    print("\nEcco le proprieta in comune dei film in ordine decrescente per influenza:\n")
    for key, value in proprieta.items():
        print(value, "\t", key)
    print("\n")


def calcola_IDF(prop):
    IDF = ''
    with open("list_idf_prop_movies", 'r') as f:  # scorre il file riga per riga
        for line in f:
            line = line.rstrip().split('\t')
            if prop == line[0]:  # quando trova la proprieta restituisce il rispettivo IDF
                IDF = line[1]
                IDF = float(IDF)
                break
    return IDF

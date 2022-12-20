from abc import abstractmethod
from typing import List

from clayrs.content_analyzer import Rank, Ratings
from clayrs.recsys import NXFullGraph, ItemNode, PropertyNode


def explain(original_ratings: Ratings, recs: Rank, file_property):

    all_explanation = {}

    for user_id in set(recs.user_id_column):

        user_explanation = []

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


        all_explanation[user_id] = user_explanation

    pass


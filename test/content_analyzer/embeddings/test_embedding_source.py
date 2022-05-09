from typing import Callable

import numpy as np
from unittest import TestCase
from math import isclose


class TestEmbeddingSource(TestCase):

    # Will be used by several test involving embeddings
    def assertWordEmbeddingMatches(self, source, embedding: np.ndarray, word: str):
        # 'similar_by_vector()' returns a list with top n
        # words similar to the vector given. I'm interested only in the most similar
        # so n = 1
        # for example, top_1 will be in the following form ("title", 1.0)
        top_1 = source.model.similar_by_vector(embedding, 1)[0]

        # So I'm using indices to access the tuples values.
        # 'like' contains how similar is 'embedding_word' to the 'embedding' vector given
        embedding_word = top_1[0]
        like = top_1[1]

        # if the word associated with the embedding vector returned by the model doesn't match the word passed as
        # argument, AssertionError is raised
        if not embedding_word == word:
            raise AssertionError("Word %s is not %s" % (embedding_word, word))

        # Obviously due to approximation the conversion won't return the
        # exact word, but if the likelihood it's equal to 1 with a maximum error of 'abs_tol'
        # I'm assuming it's exactly that word
        if not isclose(like, 1, abs_tol=1e-6):
            raise AssertionError("Word %s and result word %s do not match" % (embedding_word, word))

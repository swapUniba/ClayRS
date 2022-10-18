from __future__ import annotations
import lzma
import os
import pickle

from clayrs.content_analyzer.content_representation.representation_container import RepresentationContainer
from clayrs.content_analyzer.content_representation.content import Content


def load_content_instance(directory: str, content_id: str, only_field_representations: dict = None) -> Content:
    """
    Loads a serialized content
    Args:
        directory: Path to the directory in which the content is stored
        content_id: ID of the content to load (its filename)
        only_field_representations: Specify exactly which representation to load for the content
            (e.g. {'Plot': 0, 'Genres': 1}). Useful for alleviating memory load

    Returns:
        content (Content)
    """
    try:
        content_filename = os.path.join(directory, '{}.xz'.format(content_id))
        with lzma.open(content_filename, "rb") as content_file:
            content = pickle.load(content_file)

        if only_field_representations is not None:
            smaller_content = Content(content_id)
            field_dict_smaller = {}
            for field, repr_id_list in only_field_representations.items():
                field_dict_smaller[field] = [content.get_field_representation(field, repr_id)
                                             for repr_id in repr_id_list]

            for field, repr_list in field_dict_smaller.items():
                ext_id_list = [id if isinstance(id, str) else None for id in only_field_representations[field]]
                field_repr_container = RepresentationContainer(repr_list, ext_id_list)
                smaller_content.append_field(field, field_repr_container)

            content = smaller_content

    except FileNotFoundError:
        content = None

    return content

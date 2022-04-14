import lzma
import os
import pickle

from orange_cb_recsys.content_analyzer.content_representation.content import Content


def load_content_instance(directory: str, content_id: str) -> Content:
    """
    Loads a serialized content
    Args:
        directory (str): Path to the directory in which the content is stored
        content_id (str): Id of the content to load

    Returns:
        content (Content)
    """
    try:
        content_filename = os.path.join(directory, '{}.xz'.format(content_id))
        with lzma.open(content_filename, "rb") as content_file:
            content = pickle.load(content_file)
    except FileNotFoundError:
        content = None

    return content

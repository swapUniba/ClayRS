import functools
from os.path import isfile, splitext, join
from os import listdir
from abc import abstractmethod, ABC
from typing import Set, List, Iterable

from orange_cb_recsys.content_analyzer import SearchIndex
from orange_cb_recsys.utils import load_content_instance
from orange_cb_recsys.utils.const import logger


class LoadedContentsInterface(ABC):
    @abstractmethod
    def get_contents_interface(self):
        raise NotImplementedError


class LoadedContentsDict(LoadedContentsInterface):

    def __init__(self, contents_path: str, contents_to_load: Set[str] = None):
        self._contents_path = contents_path
        self._available_items_set = {splitext(filename)[0]
                                     for filename in listdir(contents_path)
                                     if isfile(join(contents_path, filename)) and splitext(filename)[1] == ".xz"}

        # we load all available items
        if contents_to_load is None:
            contents_to_load = self._available_items_set

        self._contents_dict = {}
        if len(contents_to_load) != 0:
            logger.info("Loading contents from disk...")
            self._contents_dict = {item_id: load_content_instance(contents_path, item_id)
                                   for item_id in contents_to_load}

            if not any(self._contents_dict.values()):
                raise FileNotFoundError(f"No contents found in {contents_path}! "
                                        f"Maybe you have misspelled the path folder?")

    def get_contents_interface(self):
        return self._contents_dict

    def get(self, key: str):
        content = self._contents_dict.get(key)
        if content is None:
            content = load_content_instance(self._contents_path, key)
            self._contents_dict[key] = content

        return content

    def get_list(self, key_list: Iterable[str]):
        contents_to_load = set(key_list) - set(self._contents_dict.keys())
        self._contents_dict.update({content: load_content_instance(self._contents_path, content)
                                    for content in contents_to_load})

        return [self._contents_dict[content] for content in key_list]

    def __getitem__(self, key: str):
        return self._contents_dict[key]

    def __iter__(self):
        yield from self._available_items_set

    def __len__(self):
        return len(self._contents_dict)

    def __str__(self):
        return str(self._contents_dict)

    def __repr__(self):
        return repr(self._contents_dict)


class LoadedContentsIndex(LoadedContentsInterface):
    def __init__(self, index_path: str):
        self._contents_index = SearchIndex(index_path)

    def get_contents_interface(self):
        return self._contents_index

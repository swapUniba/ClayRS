from os.path import isfile, splitext, join
from os import listdir
from abc import abstractmethod, ABC
from typing import Set

from orange_cb_recsys.content_analyzer import SearchIndex
from orange_cb_recsys.utils import load_content_instance
from orange_cb_recsys.utils.const import logger


class LoadedContentsInterface(ABC):
    @abstractmethod
    def get_contents_interface(self):
        raise NotImplementedError


class LoadedContentsDict(LoadedContentsInterface):

    def __init__(self, contents_path: str, contents_to_load: Set[str] = None):
        if contents_to_load is None:
            contents_to_load = {splitext(filename)[0]
                                for filename in listdir(contents_path)
                                if isfile(join(contents_path, filename)) and splitext(filename)[1] == ".xz"}

        logger.info("Loading contents from disk...")
        self._contents_dict = {item_id: load_content_instance(contents_path, item_id) for item_id in contents_to_load}

    def get_contents_interface(self):
        return self._contents_dict

    def get(self, key: str):
        return self._contents_dict.get(key)

    def __getitem__(self, key: str):
        return self._contents_dict[key]

    def __iter__(self):
        yield from self._contents_dict

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

import pandas as pd
from typing import List, Any, Union, Iterator, Dict


class RepresentationContainer:
    """
    Class that stores a generic representation. This is used in the project for storing the representations and
    ids for both the field and exogenous representations of the contents. In order to store the data, the class handles
    a dataframe with 1 column ('representation') and 2 indexes ('internal_id' and 'external_id').
    The dataframe is in the following form

                                            representation
                    internal_id external_id
                    0           'test'      FieldRepresentation instance
                    1           NaN         FieldRepresentation instance
                    2           'test2'     FieldRepresentation instance


    The index 'internal_id' is used to store the id automatically assigned by the framework to the representation.
    This default id is an integer and the column will always be in the form: [0, 1, 2, 3, ...]

    The index 'external_id' is used to store the optional id that the user can assign to the representation.
    If the user didn't define an external_id for the representation it will be set to NaN and it will be
    accessible using the internal_id only.
    The 'external_id' is a string and the column will always be in the form: ['test', NaN, 'test2', 'test3', ...]

    Both 'internal_id' and 'external_id' contain unique values, meaning that there can't be duplicates in the
    columns (except for NaN)

    The column 'representation' is used to store the instances of the representations for the content,
    It can store both instances of FieldRepresentation or ExogenousRepresentation (not both at the same time obviously,
    the column has to contain FieldRepresentation or ExogenousRepresentation only)
    By using an index value either integer (referring to 'internal_id') or string (referring to 'external_id') it's
    possible to access the corresponding representation

    Args:
        external_id_list (Union[List[Union[str, None]], Union[str, None]]): list containing the user defined ids for the
            representations, the None value is used for representations the user didn't assign an id to. It's also
            possible to pass a single value instead of a list.
        representation_list (Union[List[Any], Any]): list containing the representations instances (so eiter
            FieldRepresentation or ExogenousRepresentation). It's also possible to pass a single value instead of a
            list.

    internal_id_list is not required as an argument because it will be automatically created by the class
    """

    __slots__ = ('__int_to_ext', '__ext_to_int', '__representation_container')

    def __init__(self, representation_list: Union[List[Any], Any] = None,
                 external_id_list: Union[List[Union[str, None]], Union[str, None]] = None):
        if external_id_list is None:
            external_id_list = []
        if representation_list is None:
            representation_list = []

        if not isinstance(external_id_list, list):
            external_id_list = [external_id_list]
        if not isinstance(representation_list, list):
            representation_list = [representation_list]

        if len(external_id_list) != len(representation_list):
            raise ValueError("Representation and external_id lists must have the same length")

        self.__int_to_ext = {int_id: ext_id for int_id, ext_id in enumerate(external_id_list)}
        self.__ext_to_int = {ext_id: int_id for int_id, ext_id in
                             zip(self.__int_to_ext.keys(), self.__int_to_ext.values())
                             if ext_id is not None}

        self.__representation_container = dict(zip(self.__int_to_ext.keys(), representation_list))

    def get_internal_index(self) -> List[int]:
        """
        Returns a list containing the values in the 'internal_id' index
        """
        return list(self.__int_to_ext.keys())

    def get_external_index(self) -> List[Union[str, None]]:
        """
        Returns a list containing the values in the 'external_id' index
        """
        return list(self.__int_to_ext.values())

    def get_representations(self) -> List[Any]:
        """
        Returns a list containing the values in the 'representations' column
        """
        return list(self.__representation_container.values())

    def append(self, representation: Union[List[Any], Any],
               external_id: Union[List[Union[str, None]], Union[str, None]]):
        """
        Method used to append a list of representations (or a single representation) and their list of
        external_ids (or a single external_id) to the main dataframe. In order to do so, a new dataframe is created
        for the arguments passed by the user and it will be appended to the original dataframe. The logic behind the
        creation of the dataframe is the same as the constructor, with only one difference: the internal_id index is
        generated from the original dataframe one (so that the internal_ids are consecutive).

        Args:
            external_id (Union[List[Union[str, None]], Union[str, None]]): list containing the user defined ids for the
                representations, the None value is used for representations the user didn't assign an id to. It's also
                possible to pass a single value instead of a list.
            representation (Union[List[Any], Any]): list containing the representations instances (so eiter
                FieldRepresentation or ExogenousRepresentation). It's also possible to pass a single value instead of a
                list.
        """
        if not isinstance(external_id, list):
            external_id = [external_id]
        if not isinstance(representation, list):
            representation = [representation]

        if len(representation) != len(external_id):
            raise ValueError("Representation and external_id lists must have the same length")

        if len(self.get_internal_index()) == 0:
            next_internal_id = 0
        else:
            next_internal_id = len(self.get_internal_index())

        new_int_to_ext_dict = {int_id: ext_id for int_id, ext_id in enumerate(external_id, start=next_internal_id)}

        self.__int_to_ext.update(new_int_to_ext_dict)
        self.__ext_to_int.update({ext_id: int_id for int_id, ext_id in zip(new_int_to_ext_dict.keys(),
                                                                           new_int_to_ext_dict.values())
                                  if ext_id is not None})

        self.__representation_container.update(dict(zip(new_int_to_ext_dict.keys(), representation)))

    def pop(self, id: Union[str, int]):
        """
        Remove a specific row from the dataframe identified by the external or internal id passed as an argument.
        The representation corresponding to the selected row is also returned (in case it's needed).

        Args:
            id(Union[str, int]): used to access the row to remove from the dataframe. If it is an integer, it means
                it refers to the internal_id index, if it is a string, it means that it refers to the external_id index

        Returns:
            removed_representation (Any): representation corresponding to the removed row
        """
        removed_representation = self[id]

        if isinstance(id, str):
            id_int = self.__ext_to_int[id]
            del self.__representation_container[id_int]
            del self.__int_to_ext[id_int]
            del self.__ext_to_int[id]
        else:
            id_str = self.__int_to_ext[id]
            del self.__representation_container[id]
            del self.__ext_to_int[id_str]
            del self.__int_to_ext[id]

        return removed_representation

    def __getitem__(self, item: Union[str, int]):
        """
        Access a specific representation using an index value. The index value can be either string or integer,
        if it is an integer, it means that it is referring to the 'internal_id' index, otherwise if it is a string,
        it means that it is referring to the 'external_id' index.

        Args:
            item (Union[str, int]): value used to refer to a specific representation by accessing the index columns
        """
        original_id = item
        try:
            if isinstance(item, str):
                item = self.__ext_to_int[item]

            return self.__representation_container[item]
        except KeyError:
            raise KeyError(f"Representation with id {original_id} not found!")

    def __iter__(self) -> Iterator[Dict]:
        for internal_ind, representation in self.__representation_container.items():
            yield {'internal_id': internal_ind, 'external_id': self.__int_to_ext[internal_ind],
                   'representation': representation}

    def __len__(self):
        return len(self.get_internal_index())

    def __eq__(self, other):
        return self.__representation_container == other.__representation_container

    def __str__(self):
        dataframe = pd.DataFrame({
            'internal_id': self.get_internal_index(),
            'external_id': self.get_external_index(),
            'representation': self.get_representations()
        })
        dataframe.set_index(['internal_id', 'external_id'], inplace=True)
        return str(dataframe)

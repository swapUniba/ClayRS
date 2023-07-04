from typing import List


def id_values_merger(id_values: List[str]):
    """
    This function is used to compact a list of ids into a unique string. This can be useful when
    there is content whose id is composed by values coming from more than one field.

    Args:
        id_values (List<str>): List containing one or more ids

    Returns:
        id_merged (str): String in which the values contained in the list given in input are
            merged
    """
    if type(id_values) == str or type(id_values) == int:
        return str(id_values)
    elif type(id_values) == list:
        id_merged = ""
        for i in range(len(id_values)):
            id_merged += str(id_values[i])
            if i != len(id_values) - 1:
                id_merged += "_"
        return id_merged
    else:
        raise TypeError("id must be an integer, a string or a list of strings and/or integer")


def id_merger(raw_content: dict, field_list: List[str]) -> str:
    """
    Function that creates the list of ids and then calls id_values_merger to create a unique id
    """
    id_values = []
    for field_name in field_list:
        id_values.append(raw_content[field_name])

    return id_values_merger(id_values)

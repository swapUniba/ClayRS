from typing import Type, List
import inspect


def get_all_implemented_classes(cls: Type) -> List[Type]:
    """
    Method that returns all implemented subclasses of a given class
    (also considering subclasses of a subclass and so on)

    Args:
        cls (Type): class from which all implemented subclasses will be extracted

    Returns:
        list of implemented classes
    """
    return [c for c in get_all_classes(cls) if not inspect.isabstract(c)]


def get_all_classes(cls: Type):
    """
    Method that retrieves all subclasses of a given class
    (also considering subclasses of a subclass and so on)

    The method calls itself to find the subclasses of each subclass

    Args:
        cls (Type): class from which all implemented subclasses will be extracted

    Returns:
        list of all subclasses
    """
    return set(cls.__subclasses__()).union(
        [sub for c in cls.__subclasses__() for sub in get_all_classes(c)])

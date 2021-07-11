from typing import Type, Set
import inspect


def get_all_implemented_subclasses(cls: Type) -> Set:
    """
    Method that retrieves all implemented subclasses of a given class
    (also considering subclasses of a subclass and so on)

    The method calls itself to find the subclasses of each subclass

    Args:
        cls (Type): class from which all implemented subclasses will be extracted

    Returns:
        set containing all of cls' implemented subclasses
    """
    return set([sub for sub in cls.__subclasses__() if not inspect.isabstract(sub)]).union(
        [sub for c in cls.__subclasses__() for sub in get_all_implemented_subclasses(c) if not inspect.isabstract(sub)])


def get_all_implemented_classes(cls: Type) -> Set:
    """
    Method that retrieves all implemented subclasses of a given class
    (also considering subclasses of a subclass and so on)

    The method calls itself to find the subclasses of each subclass

    If the class passed as argument is not abstract, it is added to the set's results

    Args:
        cls (Type): class from which all implemented subclasses will be extracted

    Returns:
        set containing all of cls' implemented subclasses and cls itself if it is not abstract
    """

    classes = get_all_implemented_subclasses(cls)

    if not inspect.isabstract(cls):
        classes.add(cls)

    return classes

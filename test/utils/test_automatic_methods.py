import inspect
import unittest

from clayrs.utils.automatic_methods import autorepr


class TestAutomaticMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # init method with only positional attributes
        class OnlyPositional:
            def __init__(self, attribute1, attribute2):
                self._repr_string = autorepr(self, inspect.currentframe())

            def __repr__(self):
                return self._repr_string

        cls.only_pos_class = OnlyPositional('formal1', 'formal2')

        # init method with only args attributes
        class OnlyArgs:
            def __init__(self, *args):
                self._repr_string = autorepr(self, inspect.currentframe())

            def __repr__(self):
                return self._repr_string

        cls.only_args_class = OnlyArgs('only_args1', 'only_args2')

        # init method with only kwargs attributes
        class OnlyKwargs:
            def __init__(self, **kwargs):
                self._repr_string = autorepr(self, inspect.currentframe())

            def __repr__(self):
                return self._repr_string

        cls.only_kwargs_class = OnlyKwargs(kwargs1='only_kwargs1', kwargs2='only_kwargs2')

        # init method with all possible attributes
        class AllPossibleArgs:
            def __init__(self, attribute1, attribute2, *args, **kwargs):
                self._repr_string = autorepr(self, inspect.currentframe())

            def __repr__(self):
                return self._repr_string

        cls.all_possible_args_class = AllPossibleArgs('formal1', 'formal2', 'args1', 'args2', 'args3',
                                                      kwargs1='kwargs_val')

    def test_autorepr(self):

        expected = "OnlyPositional(attribute1='formal1', attribute2='formal2')"
        result = repr(self.only_pos_class)

        self.assertEqual(expected, result)

        expected = "OnlyArgs(*args='only_args1', *args='only_args2')"
        result = repr(self.only_args_class)

        self.assertEqual(expected, result)

        expected = "OnlyKwargs(*kwargs_kwargs1='only_kwargs1', *kwargs_kwargs2='only_kwargs2')"
        result = repr(self.only_kwargs_class)

        self.assertEqual(expected, result)

        expected = "AllPossibleArgs(attribute1='formal1', attribute2='formal2', *args='args1', " \
                   "*args='args2', *args='args3', *kwargs_kwargs1='kwargs_val')"
        result = repr(self.all_possible_args_class)

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()

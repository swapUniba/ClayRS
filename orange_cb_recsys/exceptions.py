class NoOutputDirectoryDefined(Exception):
    """
    Exception raised when an output directory has to be defined for a module in the script file but no output directory
    is found
    """
    pass


class InvalidFilePath(Exception):
    """
    Exception raised when a file path defined in the script file is not valid
    """
    pass


class ParametersError(Exception):
    """
    Exception raised for any error regarding the parameters passed by the user in the script file for a method or class
    """
    pass


class ScriptConfigurationError(Exception):
    """
    Exception raised when the basic configuration of the script file encounters an error (for example, if no module
    parameter is defined)
    """
    pass

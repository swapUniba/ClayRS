import os


def get_valid_filename(output_directory: str, filename: str, extension: str, overwrite: bool):
    """
    Method which gets a valid `filename.extension` based on the overwrite parameter

    If `overwrite=False` and there are existent file named as `filename.extension` in the output directory, the method
    checks if there are also files named as `filename (1).extension`, `filename (2).extension`,
    `filename (3).extension`, etc.
    It stops at the first non-existent `filename (x).extension` in the output directory specified
    and will return `filename (x).extension` as string

    Args:
        output_directory: Directory where the file will be saved
        filename: Name of the file to save
        extension: Extension of the file to save
        overwrite: Specify if the file to save should overwrite another file with
            same name if present. If `True` then simply `filename.extension` will be returned

    Returns:
        A valid `filename.extension` string based on the overwrite parameter
    """
    filename_try = "{}.{}".format(filename, extension)

    if overwrite is False:
        i = 0
        while os.path.isfile(os.path.join(output_directory, filename_try)):
            i += 1
            filename_try = "{} ({}).{}".format(filename, i, extension)

    return filename_try


def get_valid_dirname(output_directory: str,
                      directory_to_save: str,
                      overwrite: bool,
                      start_from_1: bool = False,
                      style: str = "parenthesis"):
    """
    Method which gets a valid directory name depending on the overwrite parameter.

    If `overwrite=False` and there is a directory named as `directory_to_save` in the `output_directory`, the method
    checks if there are also directories named as `directory_to_save (1)`, `directory_to_save (2)`,
    `directory_to_save (3)`. etc.
    It stops at the first non-existent `directory_to_save (x)` in the `output_directory` specified
    and will return `directory_to_save (x)` as string

    There are two styles supported:

    * `parenthesis` (default) -> will produce strings of the type 'directory_to_save (1)', directory_to_save (2), etc.
    * `underscore` -> will produce strings of the type 'directory_to_save_1', directory_to_save_2, etc.


    Args:
        output_directory: Directory where the `directory_to_save` will be created
        directory_to_save: Name of the directory to save
        overwrite: Specifies if the directory to save should overwrite another directory with
            same name if present.
        start_from_1: Specifies if, regardless of any other parameter, the string to return should contain (1) or '_1'
            depending on the `style` parameter chosen even if there are no other directories with same name
        style: can be `parenthesis` or `underscore`:

            * `parenthesis` (default) -> will produce strings of the type 'directory_to_save (1)',
                directory_to_save (2), etc.
            * `underscore` -> will produce strings of the type 'directory_to_save_1',
                directory_to_save_2, etc.

    Returns:
        A valid directory name

    Raises:
        `ValueError` exception if `style` different from 'parenthesis' or 'underscore'
    """
    valid_styles = {'parenthesis', 'underscore'}
    if style.lower() not in valid_styles:
        raise ValueError(f"Style {style} not supported! Only {valid_styles} are supported")

    dirname_try = directory_to_save
    if start_from_1 and style == "underscore":
        dirname_try = f"{directory_to_save}_1"
    elif start_from_1 and style == "parenthesis":
        dirname_try = f"{directory_to_save} (1)"

    if overwrite is False:
        i = 0
        while os.path.isdir(os.path.join(output_directory, dirname_try)):
            i += 1
            if style == "underscore":
                dirname_try = "{}_{}".format(directory_to_save, i)
            else:
                dirname_try = "{} ({})".format(directory_to_save, i)

    return dirname_try

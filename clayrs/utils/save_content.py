import os


def get_valid_filename(output_directory: str, filename: str, format: str, overwrite: bool):
    """
    Method which gets a valid *filename.format* based on the overwrite parameter

    If overwrite=False and there are existent file named as *filename.format* in the output directory, the method
    checks if there are also files named as *filename (1).format*, *filename (2).format*, *filename (3).format*.
    It stops at the first non existent *filename (x).format* in the output directory specified in the constructor.
    Args:
        filename (str): Name of the file to save
        format (str): Format of the file to save

    Returns:
        A valid 'filename.format' string based on the overwrite parameter
    """
    filename_try = "{}.{}".format(filename, format)

    if overwrite is False:
        i = 0
        while os.path.isfile(os.path.join(output_directory, filename_try)):
            i += 1
            filename_try = "{} ({}).{}".format(filename, i, format)

    return filename_try

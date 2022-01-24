import os


def get_valid_filename(output_directory: str, filename: str, format: str, overwrite: bool):
    filename_try = "{}.{}".format(filename, format)

    if overwrite is False:
        i = 0
        while os.path.isfile(os.path.join(output_directory, filename_try)):
            i += 1
            filename_try = "{} ({}).{}".format(filename, i, format)

    return filename_try

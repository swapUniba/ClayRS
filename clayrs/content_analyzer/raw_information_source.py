import csv
import os
from abc import ABC, abstractmethod

import json
from typing import Dict, Iterator

import mysql.connector


class RawInformationSource(ABC):
    """
    Abstract Class that generalizes the acquisition of raw descriptions of the contents
    from one of the possible raw sources.

    Args:
        encoding: define the type of encoding of data stored in the source (example: "utf-8")
    """

    def __init__(self, file_path: str, encoding: str):
        self.__file_path = file_path
        self.__encoding = encoding

    @property
    def encoding(self):
        return self.__encoding

    @property
    def file_path(self):
        return self.__file_path

    @property
    @abstractmethod
    def representative_name(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, str]]:
        """
        Iter on contents in the source, each iteration returns a dict representing a "row" in the raw content
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class DATFile(RawInformationSource):
    """
    Wrapper for a DAT file. This class is able to read from a DAT file where each entry is separated by the `::` string.
    Since a DAT file has no header, each entry can be referenced with a string representing its positional index
    (e.g. '0' for entry in the first position, '1' for the entry in the second position, etc.)

    You can iterate over the whole content of the raw source with a simple for loop: each row will be returned as a
    dictionary where keys are strings representing the positional indices, values are the entries

    Examples:
        Consider the following DAT file
        ```
        10::worker::75011
        11::without occupation::76112
        ```

        >>> file = DATFile(dat_path)
        >>> print(list(file))
        [{'0': '10', '1': 'worker', '2': '75011'},
        {'0': '11', '1': 'without occupation', '2': '76112'}]

    Args:
        file_path: path of the dat file
        encoding: define the type of encoding of data stored in the source (example: "utf-8")
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__(file_path, encoding)

    @property
    def representative_name(self) -> str:
        """
        Method which returns a meaningful name for the raw source.

        In this case it's simply the file name + its extension

        Returns:
            The representative name for the raw source
        """
        # file name with extension
        file_name = os.path.basename(self.file_path)

        return file_name

    def __iter__(self) -> Iterator[Dict[str, str]]:
        with open(self.file_path, encoding=self.encoding) as f:
            for line in f:
                line_dict = {}
                fields = line.split('::')
                for i, field in enumerate(fields):
                    field = field.strip("\n\t\r")
                    line_dict[str(i)] = field

                yield line_dict

    def __len__(self):
        with open(self.file_path, newline='', encoding=self.encoding) as dat_file:
            total_length = sum(1 for _ in dat_file)

            return total_length

    def __str__(self):
        return "DATFile"

    def __repr__(self):
        return f'DATFile(encoding={self.__encoding}, file_path={self.file_path})'


class JSONFile(RawInformationSource):
    """
    Wrapper for a JSON file. This class is able to read from a JSON file where each "row" is a dictionary-like object
    inside a list

    You can iterate over the whole content of the raw source with a simple for loop: each row will be returned as a
    dictionary

    Examples:
        Consider the following JSON file
        ```
        [{"Title":"Jumanji","Year":"1995"},
         {"Title":"Toy Story","Year":"1995"}]
        ```

        >>> file = JSONFile(json_path)
        >>> print(list(file))
        [{'Title': 'Jumanji', 'Year': '1995'},
         {'Title': 'Toy Story', 'Year': '1995'}]

    Args:
        file_path: path of the dat file
        encoding: define the type of encoding of data stored in the source (example: "utf-8")
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__(file_path, encoding)

    @property
    def representative_name(self) -> str:
        """
        Method which returns a meaningful name for the raw source.

        In this case it's simply the file name + its extension

        Returns:
            The representative name for the raw source
        """
        # file name with extension
        file_name = os.path.basename(self.file_path)

        return file_name

    def __iter__(self) -> Iterator[Dict[str, str]]:
        with open(self.file_path, encoding=self.encoding) as j:
            all_lines = json.load(j, parse_int=str, parse_float=str)
            for line in all_lines:
                yield line

    def __len__(self):
        with open(self.file_path, encoding=self.encoding) as j:
            return len(json.load(j))

    def __str__(self):
        return "JSONFile"

    def __repr__(self):
        return f'JSONFile(encoding={self.__encoding}, file_path={self.file_path})'


class CSVFile(RawInformationSource):
    r"""
    Wrapper for a CSV file. This class is able to read from a CSV file where each entry is separated by the a certain
    separator (`,` by default). So by using this class you can also read TSV file for examples, by specifying
    `separator='\t'`.

    A CSV File most typically has a header: in this case, each entry can be referenced with its column header.
    In case the CSV File hasn't a header, simply specify `has_header=False`: in this case, each entry can be referenced
    with a string representing its positional index
    (e.g. '0' for entry in the first position, '1' for the entry in the second position, etc.)

    You can iterate over the whole content of the raw source with a simple for loop: each row will be returned as a
    dictionary where keys are strings representing the positional indices, values are the entries

    Examples:

        Consider the following CSV file with header
        ```
        movie_id,movie_title,release_year
        1,Jumanji,1995
        2,Toy Story,1995
        ```

        >>> file = CSVFile(csv_path)
        >>> print(list(file))
        [{'movie_id': '1', 'movie_title': 'Jumanji', 'release_year': '1995'},
        {'movie_id': '2', 'movie_title': 'Toy Story', 'release_year': '1995'}]

        Consider the following TSV file with no header
        ```
        1   Jumanji 1995
        2   Toy Story   1995
        ```

        >>> file = CSVFile(tsv_path, separator='\t', has_header=False)
        >>> print(list(file))
        [{'0': '1', '1': 'Jumanji', '2': '1995'},
        {'0': '2', '1': 'Toy Story', '2': '1995'}]

    Args:
        file_path: Path of the dat file
        separator: Character which separates each entry. By default is a comma (`,`), but in case you need to read from
            a TSV file simply change this parameter to `\t`
        has_header: Boolean value which specifies if the file has an header or not. Default is True
        encoding: Define the type of encoding of data stored in the source (example: "utf-8")
    """

    def __init__(self, file_path: str, separator: str = ',', has_header: bool = True, encoding: str = "utf-8-sig"):
        super().__init__(file_path, encoding)
        self.__has_header = has_header
        self.__separator = separator

    @property
    def representative_name(self) -> str:
        """
        Method which returns a meaningful name for the raw source.

        In this case it's simply the file name + its extension

        Returns:
            The representative name for the raw source
        """
        # file name with extension
        file_name = os.path.basename(self.file_path)

        return file_name

    def __iter__(self) -> Iterator[Dict[str, str]]:
        with open(self.file_path, newline='', encoding=self.encoding) as csv_file:
            if self.__has_header:
                reader = csv.DictReader(csv_file, quoting=csv.QUOTE_MINIMAL, delimiter=self.__separator)
            else:
                reader = csv.DictReader(csv_file, quoting=csv.QUOTE_MINIMAL, delimiter=self.__separator)
                reader.fieldnames = [str(i) for i in range(len(reader.fieldnames))]
                csv_file.seek(0)

            yield from reader

    def __len__(self):
        with open(self.file_path, newline='', encoding=self.encoding) as csv_file:
            total_length = sum(1 for _ in csv_file)
            if self.__has_header:
                total_length -= 1

            return total_length

    def __str__(self):
        return "CSVFile"

    def __repr__(self):
        return f'CSVFile(file_path={self.file_path}, separator={self.__separator}, has_header={self.__has_header}, ' \
               f'encoding={self.encoding})'


class SQLDatabase(RawInformationSource):
    """
    Wrapper for a SQL database.

    You can iterate over the whole content of the raw source with a simple for loop: each row will be returned as a
    dictionary where keys are strings representing the positional indices, values are the entries

    Examples:

        Consider the following SQL table for the databaase 'movies' in localhost
        ```
        +----------+-------------+--------------+
        | Movie ID | Movie Title | Release Year |
        +----------+-------------+--------------+
        |        1 | Jumanji     |         1995 |
        |        2 | Toy Story   |         1995 |
        +----------+-------------+--------------+
        ```

        >>> file = SQLDatabase(host='127.0.0.1', username='root', password='root',
        >>>                    database_name='movies', table_name='movies_table')
        >>> print(list(file))
        [{'Movie ID': '1', 'Movie Title': 'Jumanji', 'Release Year': '1995'},
        {'Movie ID': '2', 'Movie Title': 'Toy Story', 'Release Year': '1995'}]

    Args:
        host: host ip of the sql server
        username: username for the access
        password: password for the access
        database_name: name of database
        table_name: name of the database table where data is stored
        encoding: Define the type of encoding of data stored in the source (example: "utf-8")
    """

    def __init__(self, host: str,
                 username: str,
                 password: str,
                 database_name: str,
                 table_name: str,
                 encoding: str = "utf-8"):
        super().__init__('', encoding)
        self.__host: str = host
        self.__username: str = username
        self.__password: str = password
        self.__database_name: str = database_name
        self.__table_name: str = table_name

        conn = mysql.connector.connect(host=self.__host,
                                       user=self.__username,
                                       password=self.__password,
                                       charset=self.encoding)
        cursor = conn.cursor()
        query = """USE """ + self.__database_name + """;"""
        cursor.execute(query)
        conn.commit()
        self.__conn = conn

    @property
    def host(self) -> str:
        return self.__host

    @property
    def username(self) -> str:
        return self.__username

    @property
    def password(self) -> str:
        return self.__password

    @property
    def database_name(self) -> str:
        return self.__database_name

    @property
    def table_name(self) -> str:
        return self.__table_name

    @property
    def conn(self):
        return self.__conn

    @host.setter
    def host(self, host: str):
        self.__host = host

    @username.setter
    def username(self, username: str):
        self.__username = username

    @password.setter
    def password(self, password: str):
        self.__password = password

    @database_name.setter
    def database_name(self, database_name: str):
        self.__database_name = database_name

    @table_name.setter
    def table_name(self, table_name: str):
        self.__table_name = table_name

    @conn.setter
    def conn(self, conn):
        self.__conn = conn

    @property
    def representative_name(self) -> str:
        """
        Method which returns a meaningful name for the raw source.

        In this case it's the host name followed by the table name

        Returns:
            The representative name for the raw source
        """
        return f"{self.host}/{self.table_name}"

    def __iter__(self) -> Iterator[Dict[str, str]]:
        cursor = self.conn.cursor(dictionary=True)
        query = """SELECT * FROM """ + self.table_name + """;"""
        cursor.execute(query)
        for result in cursor:
            yield result

    def __len__(self):
        cursor = self.conn.cursor()
        query = """SELECT * FROM """ + self.table_name + """;"""
        cursor.execute(query)
        cursor.fetchall()

        return cursor.rowcount

    def __str__(self):
        return "SQLDatabase"

    def __repr__(self):
        return f'SQLDatabase(host={self.__host},' \
               f'username={self.__username}, password={self.__password}, ' \
               f'database_name={self.__database_name}, table_name={self.__table_name}, encoding={self.encoding})'

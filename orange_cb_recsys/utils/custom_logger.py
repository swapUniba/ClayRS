import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    def __init__(self, format: str):
        super().__init__()

        grey = "\x1b[38;21m"
        yellow = "\u001b[33m"
        red = "\u001b[31m"
        bold_red = "\u001b[1m\u001b[31m"
        reset = "\u001b[0m"
        header = "\u001b[32m"

        self.FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: reset + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(CustomFormatter("\r%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"))

        self.addHandler(handler)

    def enable(self, level=logging.INFO):
        self.setLevel(level)

    def disable(self, level=logging.CRITICAL):
        self.setLevel(level)

    def get_logger(self):
        return self

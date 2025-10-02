import logging
from pathlib import Path


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    white = "\x1b[37m"
    yellow = "\x1b[93m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once (Singleton pattern)
        if CustomLogger._initialized:
            return

        CustomLogger._initialized = True

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())

        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(ch)

        # Prevent logging from propagating to the root logger
        self.logger.propagate = False

    def get_logger(self, name=None):
        """
        Get a logger instance with an optional name.
        If name is provided, it will be a child of the main logger.
        """
        if name:
            return self.logger.getChild(name)
        return self.logger

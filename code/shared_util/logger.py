import logging


class Logger:
    def __init__(self, logging_level):
        logging_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging_level, format=logging_format)
        self.logger = logging.getLogger(__name__)

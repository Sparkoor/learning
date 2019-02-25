import logging
from termcolor import *


class Logging:
    def __init__(self):
        logFormat = "%(asctime)s- %(levelname)s-%(lineno)d:%(message)s"
        logging.basicConfig(level=logging.INFO, format=logFormat)

    def info(self, message):
        logging.info(colored(message, 'yellow'))


if __name__ == "__main__":
    logger = Logging()
    logger.info("aaa")

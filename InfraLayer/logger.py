import os.path
import logging
import logging.config

class Logging:

    __instance = None
    logger = None

    @staticmethod
    def getInstance():
        if Logging.__instance == None:
            Logging()
        return Logging.__instance

    def __init__(self):
        if Logging.__instance != None:
            raise Exception("Esta es una clase Singleton!!! Por Dios!")
        else:
            Logging.__instance = self

            logging.config.fileConfig('../config.ini')
            self.logger = logging.getLogger('sLogger')

    def debug(self, deb):
        self.logger.debug(deb)

    def info(self, inf):
        self.logger.info(inf)

    def warn(self, war):
        self.logger.warn(war)

    def error(self, err):
        self.logger.error(err)

    def critical(self, cri):
        self.logger.critical(cri)
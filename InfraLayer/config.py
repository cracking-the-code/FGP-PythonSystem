import os.path
import configparser

class Config:

    config = None

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('../config.ini')

    def conf(self):
        return self.config
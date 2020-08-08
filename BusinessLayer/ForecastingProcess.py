import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config
from DataLayer import DbConnection
import pandas as pd

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()

class forecastingProcess(object):

    def __init__(self):
        pass

    def onlineAnalysis():
        pass
    
    def Train(self, model):
        """GENERAL TRAINING METHOD"""
        logg.info(f"TrainProcess[Train]: The Trainning will execute the method: {model.modelName}")
        algorithm = TrainProcess.__getModel(model.modelName,model)
        data = algorithm.Train()

        return data #It returns a trained model
    
    def Predict(model):
        """GENERAL PREDICT METHOD"""
        return None

    def __getModel(modelType, model):
        if(modelType == "ARIMA"):
            logg.info("TrainProcess[__getModel]: ARIMA model was selected")
            arimaModel = ARIMA_Model(model)
            return arimaModel
        elif(modelType == "RNN"):
            logg.info("TrainProcess[__getModel]: RNN model was selected")
            rnnModel = RNN_Model(model)
            return rnnModel
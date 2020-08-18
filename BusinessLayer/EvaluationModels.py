import sys
sys.path.append("..")

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from math import sqrt
import pandas as pd
import numpy as np

from InfraLayer.logger import Logging
from InfraLayer.config import Config

class Evaluation(object):

    def __init__(self):
        pass

    def graficar(self, DataFrame):
        plt.plot(DataFrame)
        plt.show()

    def graficarEntrenamientoPrediccionTotal():
        pass

    def graficarEntrenamientoPrediccion():
        pass

    def graficarTestingPrediccion(self, dfTesting, dfPrediccion):
        
        plt.plot(dfTesting, color = 'red', label = 'Datos de Prueba')
        plt.plot(dfPrediccion, "--", color = 'blue', label = 'Datos de Prediccion')
        
        plt.show()
              

    def plotting(real, dataTest, prediction):
        values = real.values
        valuesTest = pd.DataFrame(dataTest)
        valuesTest = valuesTest.values
        forecast = np.concatenate((dataTest, prediction))
        plt.plot(real, color = 'red', label = 'Serie Temporal de Potencia')
        plt.xlabel('Tiempo')
        plt.ylabel('Potencia')
        plt.show()
        plt.plot(forecast, "--", linewidth = 2, color = 'red', label = 'Prediccion')
        plt.plot(valuesTest, color = 'blue', label = 'Entrenamiento')
        plt.xlabel('Tiempo')
        plt.ylabel('Potencia')
        plt.legend()
        plt.show()
    
    def metrics(expected, predicted):
        mae = mean_absolute_error(expected, predicted)
        mse = mean_squared_error(expected, predicted)
        rmse = sqrt(mse)
    
        return mae, mse, rmse
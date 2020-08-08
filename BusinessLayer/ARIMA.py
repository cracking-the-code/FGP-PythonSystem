import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config

from Temporabilidad import Temporabilidad

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()

import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima
from datetime import datetime

class ARIMA_Model(object):

    def __init__(self, dataFrame, **context):

        try:
            logg.info("ARIMA_Model[__init__]: El modelo ARIMA sera creado")

            self.__df = dataFrame
            self.__periodo = context['periodo']
            self.__steps = context['maxSteps']
            self.__periodFormat = context['periodFormat']
            self.__temporabilidad = Temporabilidad().getTemporabilidad(context['periodFormat'])
            self.__freq = Temporabilidad().getFreq(context['periodFormat'],context['periodo'])
        
        except Exception as ex:
            logg.error("ARIMA_Model[__init__]: El modelo ARIMA no pudo ser creado")
            logg.error(f"ARIMA[__init__]: Error: {str(ex.args)}")

    def Train(self):

        try:

            logg.info("ARIMA[Train]: Comienza el proceso de entrenamiento")

            self.__df = self.__df.sort_values(by=['TimeMeasure'],ascending=True)
            lastTime = self.__df.tail(1).index.item()

            newDate = Temporabilidad().getNewDate(lastTime, self.__periodFormat, self.__periodo)
            index = pd.date_range(newDate, periods=self.__steps, freq=self.__freq)
            
            logg.info(f"ARIMA[Train]: NewDate: {str(newDate)}, lastTime: {str(lastTime)}")

            forecasting_DataFrame = pd.DataFrame(columns=self.__df.columns, index=index)

            
            for column in range(len(self.__df.iloc[0])):
                measure = self.__df.iloc[0:,column]
                predictionMeasure = self.__trainColumn(measure)
                forecasting_DataFrame[measure.name] = predictionMeasure

            return forecasting_DataFrame

        except Exception as ex:
            
            logg.error("ARIMA[Train]: Un Error ha ocurrido durante el proceso de entrenamiento y prediccion")
            logg.error("ARIMA[Train]: Error: " + str(ex.args))


    def __trainColumn(self, data):
        
        try:

            logg.info("ARIMA[Train]: El valor: " + data.name + " Sera entrenado. Hasta aqui mi reporte Joquin :)")

            arima = auto_arima(data.iloc[0:,], m = self.__periodo)
            parametros = arima.order
            p = parametros[0]
            d = parametros[1]
            q = parametros[2]

            modelo = ARIMA(data.iloc[0:,], order=(p,d,q))
            results = modelo.fit()

            logg.info(f"ARIMA[Train]: El valor: {data.name} ha sido entrenado exitosamente!!!!")
            logg.info(f"ARIMA[Train]: El valor: {data.name} sera predecido :O")

            predictedResults = results.predict(start=len(data), end=len(data)+self.__steps-1,typ='levels')

            logg.info(f"ARIMA[Train]: El valor: {data.name} ha sido predecido exitosamente!!!!")

            return predictedResults

        except Exception as ex:

            logg.error("ARIMA[Train]: Ha ocurrido un error durante el proceso de prediccion")
            logg.error(f"ARIMA[Train]: {str(ex.args)}")
            return None


    def Predict(self):
        return None

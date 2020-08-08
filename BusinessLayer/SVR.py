import sys
sys.path.append("..")

from Temporabilidad import Temporabilidad

from InfraLayer.logger import Logging
from InfraLayer.config import Config

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from math import sqrt

import pandas as pd
import numpy as np

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()

class SVR_Model(object):

    def __init__(self, dataFrame, **context):

        try:
            logg.info("SVR_Model[__init__]: El modelo SVR sera creado")

            self.__df = dataFrame
            self.__periodo = context['periodo']
            self.__steps = context['maxSteps']
            self.__periodFormat = context['periodFormat']
            self.__temporabilidad = Temporabilidad().getTemporabilidad(context['periodFormat'])
            self.__freq = Temporabilidad().getFreq(context['periodFormat'],context['periodo'])
            self.__trainPorcentaje = context['trainPorcentaje']
            self.__window = context['window']
            self.__isTrain = context['isTrain']

        except Exception as ex:
            logg.error("SVR_Model[__init__]: El modelo SVR no pudo ser creado")
            logg.error(f"SVR_Model[__init__]: Error: {str(ex.args)}")

    def Train(self):
        
        dataSet = None

        if self.__isTrain:
            dataSet, test = SVR_Model.__split_timeSeries(self.__df, self.__trainPorcentaje)
        else:
            dataSet, test = SVR_Model.__split_timeSeries(self.__df, 90)

        # Se Convierte el DataFrame a un Problema de Aprendizaje Supervisado
        reframed = SVR_Model.__series_to_supervised(dataSet, round(self.__window), 1)
        # Se Toman los pasos anteriores para entrenar
        stepsBack, trainn = SVR_Model.__steps_back(reframed)
        # Se Entrena al modelo
        results = SVR_Model.__predictFutureSeries(stepsBack,trainn,self.__window)
        
        return results


    def __split_timeSeries(data, porcentaje):

        logg.info(f"SVR_Model[__split_timeSeries]: El dataFrame tendra un {porcentaje}% de datos de Entrenamiento")

        length = data.shape[0]
        ntime = round(length * (porcentaje / 100))
        values = data.values
        train = values[:ntime, :]
        test = values[ntime:, :]

        return train, test

    def __series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

        logg.info(f"SVR_Model[series_to_supervised]: El dataset se convierte aun problema de aprendizaje supervisado")
        
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def __steps_back(data):

        steps = data.shape[1] - 1
        values = data.values
        
        logg.info("SVR_Model[__steps_back]: Se toman %d pasos anteriores de la serie temporal " % steps)
        
        train = values[(steps):,:]
        x_test = train[:,:-1]
        y_test = train[:,-1]
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]))

        return x_test, y_test

    def __predictFutureSeries(x_test, x_train, WINDOW):
        results = []
        for i in range(WINDOW):
            logg.info("SVR_Model[__predictFutureSeries]: CICLO: %d" % i)
            parcial = SVR_Model.__trainModel(x_test, x_train)
            results.append(parcial[0])
            x_test = SVR_Model.__addNewValue(x_test,parcial[0])

        return results

    def __trainModel(train_x, train_y):
        logg.info("SVR_Model[__trainModel]: Se procede a realizar el entrenamiento con el modelo SVR")
        param_grid = {
                        "C": np.linspace(10**(-2),10**3,100),
                        "gamma": np.linspace(0.0001,1,20)
                     }
        
        mod = SVR(epsilon = 0.1, kernel = 'rbf')
        model = GridSearchCV(estimator = mod, param_grid = param_grid,scoring = "neg_mean_squared_error", verbose = 0)
    
        scalerIn = MinMaxScaler(feature_range=(-1,1))
        scalerOut = MinMaxScaler(feature_range=(-1,1))
    
        scaledTrain = scalerIn.fit_transform(train_x)
        scaledTrainFuture = scalerOut.fit_transform(train_y.reshape(-1,1))
        best_model = model.fit(scaledTrain, scaledTrainFuture.ravel())
    
        logg.info("SVR_Model[__trainModel]: Se procede a realizar la prediccion con el modelo SVR")
        predicted_tr = model.predict(scaledTrain)
        predicted_tr = scalerOut.inverse_transform(predicted_tr.reshape(-1,1))
    
        return predicted_tr

    def __addNewValue(x_test, newValue):
        for i in range(x_test.shape[1]-1):
            x_test[0][i] = x_test[0][i+1]

        x_test[0][x_test.shape[1]-1]=newValue

        return x_test
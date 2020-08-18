import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config

from DataLayer import DbConnection
from BusinessLayer import DataProcess
from EvaluationModels import Evaluation
from ARIMA import ARIMA_Model
from SVR import SVR_Model
import pandas as pd
import numpy as np

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()
db      = DbConnection.DbConnection.getInstance()
dp      = DataProcess.DataProcess()

if __name__ == "__main__":
    
    #   El diccionario context especifica los parametros de configuracion para los modelos:
    #   ARIMA & SVR
    context = {
        'idDev'           : 'Joshua001',
        'variable'        : 'Potency',
        'fechaIni'        : '2020-03-15 00:00:00',
        'fechaFin'        : '2020-03-16 00:00:00',
        'periodFormat'    : 'HOUR',
        'periodo'         : 1,
        'maxSteps'        : 3,
        'trainPorcentaje' : 50,
        'window'          : 5,
        'isTrain'         : True
    }
    
    #   Variables tomadas del diccionario context para ser usadas en la clase de DataProcess
    idDev           = context['idDev']   
    variable        = context['variable']
    fechaIni        = context['fechaIni']
    fechaFin        = context['fechaFin']
    periodFormat    = context['periodFormat']
    periodo         = context['periodo']
    

    #   Se establece la conexion a la Base de Datos
    db.connect()

    queryData           = dp.getMediciones(idDev,variable,fechaIni, fechaFin)
    dataProcessed       = dp.processData(queryData, periodFormat, periodo)

    train, test         = dp.splitDatos(dataProcessed, 50)
    

    #windows             = dp.ventanaDeslizante(dataProcessed,1,5)

    #dp.trainingTestingData(windows,5)

    #context['maxSteps'] = test.shape[0]
    #context['window']   = test.shape[0]


    #print('VAMOA Entrenars en ARIMA')
    #model = ARIMA_Model(train, **context)
    #print('VAMOA Entrenars en ARIMA X2')
    #results = model.Train()




    #print(dataProcessed)

    print('VAMOA Entrenars en SVR')
    model = SVR_Model(dataProcessed, **context)
    print('VAMOA Entrenars en SVR X2')
    results = model.Train()



    #evaluation = Evaluation()

    #evaluation.graficarTestingPrediccion(test,results)
    #evaluation.graficar(results)
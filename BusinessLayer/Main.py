import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config

from DataLayer import DbConnection
from BusinessLayer import DataProcess
from ARIMA import ARIMA_Model
from SVR import SVR_Model

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()
db      = DbConnection.DbConnection.getInstance()
dp      = DataProcess.DataProcess()

if __name__ == "__main__":
    
    context = {
        'idDev'           : 'Joshua001',
        'variable'        : 'Potency',
        'fechaIni'        : '2020-03-15 00:00:00',
        'fechaFin'        : '2020-03-16 00:00:00',
        'periodFormat'    : 'HOUR',
        'periodo'         : 1,
        'maxSteps'        : 3,
        'trainPorcentaje' : 90,
        'window'          : 5,
        'isTrain'         : True
    }
    
    idDev           = context['idDev']      
    variable        = context['variable']   
    fechaIni        = context['fechaIni']
    fechaFin        = context['fechaFin']
    periodFormat    = context['periodFormat']
    periodo         = context['periodo']
    
    db.connect()

    queryData       = dp.getMediciones(idDev,variable,fechaIni, fechaFin)
    dataProcessed   = dp.processData(queryData, periodFormat, periodo)

    #print('VAMOA Entrenars en ARIMA')
    #model = ARIMA_Model(dataProcessed, **context)
    #print('VAMOA Entrenars en ARIMA X2')
    #model.Train()

    print(dataProcessed)

    print('VAMOA Entrenars en SVR')
    model = SVR_Model(dataProcessed, **context)
    print('VAMOA Entrenars en SVR X2')

    model.Train()

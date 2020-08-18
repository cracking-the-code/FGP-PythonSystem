import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config
from DataLayer import DbConnection
import pandas as pd

cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()
db      = DbConnection.DbConnection.getInstance()

class DataProcess(object):

    def __init__(self):
        pass

    def getMediciones(self, idDispositivo, variable, fechaI, fechaF):
        
        try:
            logg.info(f"DataProcess[getMediciones]: Se consultaran las mediciones para: {idDispositivo}")

            df = db.getMedicion(idDispositivo, variable, fechaI, fechaF)
            
            logg.info("DataProcess[getMediciones]: Se ha realizado la consulta!")

            return df

        except Exception as ex:

            logg.error("DataProcess[getMediciones]: Un error ha ocurrido durante la consulta")
            logg.error(f"DataProcess[getMediciones]: Error: {str(ex.args)}")
            
            return None

    def processData(self, data, periodFormat, periocidad):

        try:
            logg.info("DataProcess[processData]: Los datos seran procesados!!!")

            dt = pd.DataFrame(data)

            resampleOption = None

            if(periodFormat == "YEAR"):
                resampleOption = str(365 * periocidad) + 'D'
            elif(periodFormat == "MONTH"):
                resampleOption = str(30 * periocidad) + 'D'
            elif(periodFormat == "WEEK"):
                resampleOption = str(7 * periocidad) + 'D'
            elif(periodFormat == "DAY"):
                resampleOption = str(periocidad) + 'D'
            elif(periodFormat == "HOUR"):
                resampleOption = str(periocidad) + 'H'
            elif(periodFormat == "MINUTES"):
                resampleOption = str(periocidad) + 'T'
            elif(periodFormat == "SECONDS"):
                resampleOption = str(periocidad) + 'S'
            elif(periodFormat == "SEASON"):
                resampleOption = '3M'
            elif(periodFormat == "BIMESTRE"):
                resampleOption = '2M'
            elif(periodFormat == "TRIMESTRE"):
                resampleOption = '3M'
            elif(periodFormat == "SEMESTRE"):
                resampleOption = '6M'

            dt = dt.resample(resampleOption).mean().ffill()

            logg.info("DataProcess[processData]: Los datos han sido procesados")

            return dt

        except Exception as ex:

            logg.error("DataProcess[processData]: Ha ocurrido un error durante el proceso")
            logg.error(f"DataProcess[processData]: Error: {str(ex.args)}")
            
            return None

    def splitDatos(self, data, porcentaje):

        logg.info(f"DataProcess[splitDatos]: Se usara el: {porcentaje}% para Entrenamiento y un: {100-porcentaje}% para Testing")

        length = data.shape[0]
        ntime = round(length * (porcentaje / 100))

        train = data.iloc[:ntime]
        test  = data.iloc[ntime:]

        return train, test

    def ventanaDeslizante(self, data, step, window, target=1, op=1):

        df = pd.DataFrame()
        window += 1


        names = list()

        for i in range(-window+target, target):
            names += [('(t%d)' % (i)) if i<0 else ('(t+%d)' % (i))]

        for i in range(0, data.shape[0], step):
            if(i+window > data.shape[0]):
                break
            lagged = data.iloc[i:i+window].reset_index(drop=True)
            lagged = lagged.T.reset_index(drop=True)

            df = df.append(lagged)
        
        df.columns = names

        print(df)

        return df

    def trainingTestingData(self, data, window_Width, target_Width=1):

        data = data.values

        train = data[:,:-target_Width]
        test = data[:,-target_Width]

        print('TRAIN')
        print(train)
        print('TEST')
        print(test)

        return train, test
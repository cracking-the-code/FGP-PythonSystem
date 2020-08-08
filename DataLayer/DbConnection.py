import sys
sys.path.append("..")

from InfraLayer.logger import Logging
from InfraLayer.config import Config

from sqlalchemy import Table, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy import MetaData
import sqlalchemy as db
import pandas as pd


cnf     = Config()
config  = cnf.conf()
logg    = Logging.getInstance()


class DbConnection(object):

    __instance = None
    __engine = None
    __conn = None
    __table = None

    @staticmethod
    def getInstance():
        if DbConnection.__instance == None:
            DbConnection()
        return DbConnection.__instance

    def __init__(self):
        if DbConnection.__instance != None:
            raise Exception("Esta es una clase Singleton!")
        else:
            DbConnection.__instance = self

    def connect(self):
        stringConnection = ("{}://{}:{}@{}/{}").format(
                            config['MariaDB']['dbMariaDriver'],
                            config['MariaDB']['dbMariaUser'],
                            config['MariaDB']['dbMariaPass'],
                            config['MariaDB']['dbMariaServer'],
                            config['MariaDB']['dbMariaDb'])

        try:
            logg.info("DbConnection[connect]: Se inicia la coneccion a la Base de Datos")
            logg.info(f"DbConnection[connect]: DbServer: {config['MariaDB']['dbMariaServer']}")
            logg.info(f"DbConnection[connect]: DbDataDase: {config['MariaDB']['dbMariaDb']}")
            logg.info(f"DbConnection[connect]: DbUser: {config['MariaDB']['dbMariaUser']}")
        
            DbConnection.__engine = db.create_engine(stringConnection)
            DbConnection.__conn = DbConnection.__engine.connect()

            meta = MetaData(DbConnection.__engine)

            self.__table = Table('Tbl_ForecastingResults', meta,
                Column('IdResult', Integer, primary_key=True),
                Column('IdOrder',String),
                Column('IdDev', String),
                Column('ProcessTime', DateTime),
                Column('Status', String),
                Column('JsonResults', String))

            logg.info("DbConnection[connect]: Coneccion Exitosa!!!")

        except Exception as ex: 
            logg.error("DbConnection[connect]: Ha surgido un error al conectarse a la Base de Datos")
            logg.error(f"DbConnection[connect]: Error: {str(ex.args)}")


    def getMedicion(self, idDev, mainVar, dateI, dateF):
        
        df = None
        query = DbConnection.__consultaMedicion(idDev, mainVar, dateI, dateF)

        try:
            logg.info("DbConnection[getMedicion]: Se realizara la siguiente Query en la BD")
            logg.info(f"DbConnection[getMedicion]: Query: {query}")

            df = pd.read_sql(query, DbConnection.__engine)
            df = df.set_index('TimeMeasure')

            logg.info("DbConnection[getMedicion]: La consulta sera convertida a un DataFrame")

        except Exception as ex:
            
            logg.error("DbConnection[getMedicion]: Ocurrio un error durante la conversion a DataFrame")
            logg.error(f"DbConnection[getMedicion]: Error: {str(ex.args)}")
            df = None

        return df


    def __consultaMedicion(idDev, mainVar, dateI, dateF):

        query = f"SELECT TimeMeasure, {mainVar} FROM Tbl_DeviceMeasurement WHERE IdDev = '{idDev}'"
        query = f"{query} AND TimeMeasure >= '{dateI}' AND TimeMeasure <= '{dateF}'"

        return query

    def insertPrediccion(self, idOrder, idDev, jsonResults):
        
        try:
        
            logg.info("DbConnection[insertPrediccion] Una prediccion sera guardada en la tabla Tbl_ForecastingResults")

            ins = self.__table.insert().values(
                IdOrder = idOrder,
                IdDev = idDev,
                ProcessTime = dt.now(),
                Status = "SUCCESSFUL",
                JsonResults = jsonResults)

            db.getConnection().execute(ins)

            logg.info("DbConnection[insertPrediccion] La prediccion ha sido guardada!!!")

        except Exception as ex:

            logg.error("DbConnection[insertPrediccion] La prediccion no pudo ser guardada")
            logg.error("DbConnection[insertPrediccion] Error: " + str(ex.args))
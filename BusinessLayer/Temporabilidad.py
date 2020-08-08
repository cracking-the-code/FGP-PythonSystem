from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

class Temporabilidad(object):

    def getTemporabilidad(self, unidad):
        if(unidad == "YEAR"):
            return 1
        elif(unidad == "MONTH"):
            return 1
        elif(unidad == "WEEK"):
            return 4
        elif(unidad == "DAY"):
            return 7
        elif(unidad == "HOUR"):
            return 24
        elif(unidad == "MINUTES"):
            return 60
        elif(unidad == "SECONDS"):
            return 60
        elif(unidad == "SEASON"):
          return 7
        elif(unidad == "HOUR"):
            return 24
        elif(unidad == "MINUTES"):
            return 60
        elif(unidad == "SECONDS"):
            return 60
        elif(unidad == "SEASON"):
            return 4
        elif(unidad == "BIMESTRE"):
            return 1
        elif(unidad == "TRIMESTRE"):
            return 1
        elif(unidad == "SEMESTRE"):
            return 1

    def getFreq(self, periodFormat):
        if(periodFormat == "YEAR"):
            return 'AS'
        elif(periodFormat == "MONTH"):
            return 'M'
        elif(periodFormat == "WEEK"):
            return 'W'
        elif(periodFormat == "DAY"):
            return 'D'
        elif(periodFormat == "HOUR"):
            return 'H'
        elif(periodFormat == "MINUTES"):
            return 'T'
        elif(periodFormat == "SECONDS"):
            return 'S'
        elif(periodFormat == "SEASON"):
            return '3M'
        elif(periodFormat == "BIMESTRE"):
            return '2M'
        elif(periodFormat == "TRIMESTRE"):
            return '3M'
        elif(periodFormat == "SEMESTRE"):
            return '6M'

    def getFreq(self, periodFormat, periocidad):
        if(periodFormat == "YEAR"):
            return str(365 * periocidad) + 'D'
        elif(periodFormat == "MONTH"):
            return str(30 * periocidad) + 'D'
        elif(periodFormat == "WEEK"):
            return str(7 * periocidad) + 'D'
        elif(periodFormat == "DAY"):
            return str(periocidad) + 'D'
        elif(periodFormat == "HOUR"):
            return str(periocidad) + 'H'
        elif(periodFormat == "MINUTES"):
            return str(periocidad) + 'T'
        elif(periodFormat == "SECONDS"):
            return str(periocidad) + 'S'
        elif(periodFormat == "SEASON"):
            return '3M'
        elif(periodFormat == "BIMESTRE"):
            return '2M'
        elif(periodFormat == "TRIMESTRE"):
            return '3M'
        elif(periodFormat == "SEMESTRE"):
            return '6M'

    def getNewDate(self, lastTime, periodFormat, steps):
        
        year    = lastTime.year
        month   = lastTime.month
        day     = lastTime.day
        hour    = lastTime.hour
        minute  = lastTime.minute
        second  = lastTime.second

        newDate = datetime(year, month, day, hour, minute, second)

        if(periodFormat == "YEAR"):
            return newDate + relativedelta(years=steps)
        elif(periodFormat == "MONTH"):
            return newDate + relativedelta(months=steps)
        elif(periodFormat == "WEEK"):
            return newDate + relativedelta(weeks=steps)
        elif(periodFormat == "DAY"):
            return newDate + timedelta(days=steps)
        elif(periodFormat == "HOUR"):
            return newDate + relativedelta(hours=steps)
        elif(periodFormat == "MINUTES"):
            return newDate + timedelta(minutes=steps)
        elif(periodFormat == "SECONDS"):
            return newDate + timedelta(seconds=steps)
        elif(periodFormat == "SEASON"):
            return newDate + relativedelta(months=3*steps)
        elif(periodFormat == "BIMESTRE"):
            return newDate + relativedelta(months=2*steps)
        elif(periodFormat == "TRIMESTRE"):
            return newDate + relativedelta(months=3*steps)
        elif(periodFormat == "SEMESTRE"):
            return newDate + relativedelta(months=6*steps)

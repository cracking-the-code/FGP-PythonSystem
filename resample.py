import pandas as pd
import numpy as np

df = pd.read_csv('20200113.csv',  parse_dates=[0], header=0,index_col=0, squeeze=True)
df.head()

def resampleSeries(data, resampleTime):
    dt = pd.DataFrame(data)
    if(resampleTime < 0.99):
        period = "%dS" % (resampleTime * 100)
        print(period)
        return dt.resample(period).mean()
    else:
        period = "%dT" % resampleTime
        return dt.resample(period).mean()

one = resampleSeries(df, 10)
print(one)
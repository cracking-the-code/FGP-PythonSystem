import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('20200113.csv',  parse_dates=[0], header=None,index_col=0, squeeze=True)
df.head()

PASOS=7

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
values = df.values
values = values.astype('float32')
values = values.reshape(-1,1)
reframed = series_to_supervised(values, 100, 2)
print(reframed)
print(reframed.describe())

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 2)
reframed.head()

print(df.index.min())
print(df.index.max())

print(len(df['2020-01-13 11']))
print(len(df['2020-01-13 12']))
print(len(df['2020-01-13 13']))
print(len(df['2020-01-13 14']))
print(len(df['2020-01-13 15']))



print(df.describe())

print(reframed)

resampli = "%dT" % 60
print(type(resampli))
print(df.resample(resampli).mean())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('january2008.csv',  parse_dates=[0], header=0,index_col=0, squeeze=True)
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

def split_timeSeries(data, percentage):
    length = data.shape[0]
    train = round(length * (percentage / 100))
    test = length - train
    print("Train: %d" % train)
    print("Test: %d" % test)
    print("Total: %d" % (train + test))
    trainDf = data[:train]
    testDf = data[train:]
    print("===================TRAIN==================================")
    print(trainDf)
    print("===================TEST===================================")
    print(testDf)
    return trainDf, testDf

def reshape(data, min=-1, max=1):
    values = data.values
    values = values.astype('float32')
    
    scaler = MinMaxScaler(feature_range=(min,max))
    
    values = values.reshape(min,max)
    scaled = scaler.fit_transform(values)

    return scaled

def trainModel(trainn, tests):
    param_grid = {
                    "C": np.linspace(10**(-2),10**3,100),
                    "gamma": np.linspace(0.0001,1,20)
                 }
    
    mod = SVR(epsilon = 0.1, kernel = 'rbf')
    model = GridSearchCV(estimator = mod, param_grid = param_grid,scoring = "neg_mean_squared_error", verbose = 0)

    values = trainn.values
    values = values.astype('float32')

    scalerIn = MinMaxScaler()
    scalerOut = MinMaxScaler()

    scaledTrain = scalerIn.fit_transform(values[:,0].reshape(-1,1))
    scaledTrain1 = scalerOut.fit_transform(values[:,1].reshape(-1,1))

    print("===================================")
    print(values)
    print("===================================")
    print(scaledTrain)    
    print("===================================")
    print(scaledTrain1)    

    value = tests.values
    value = value.astype('float32')

    scaledTest = scalerIn.fit_transform(value[:,0].reshape(-1,1))
    scaledTest1 = scalerOut.fit_transform(value[:,1].reshape(-1,1))

    print("===================================")
    print(value)
    print("===================================")
    print(scaledTest)    
    print("===================================")
    print(scaledTest1)    

    best_model = model.fit(scaledTrain, scaledTrain1.ravel())

    #prediction
    predicted_tr = model.predict(scaledTrain)
    predicted_te = model.predict(scaledTest)

    # inverse_transform because prediction is done on scaled inputs
    predicted_tr = scalerOut.inverse_transform(predicted_tr.reshape(-1,1))
    predicted_te = scalerOut.inverse_transform(predicted_te.reshape(-1,1))

    return predicted_tr, predicted_te

resampled = resampleSeries(df, 720)
print(resampled)

# load dataset
#values = reshape(resampled)
reframed = series_to_supervised(resampled, 1, 1)
print(reframed)
print(reframed.describe())

print(reframed.shape[0])
train, test = split_timeSeries(reframed, 49)

print("train: %d" % len(train))
print("test: %d" % len(test))

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
valueTr = train.values
valueTr = valueTr.astype('float32')

valueTs = train.values
valueTs = valueTs.astype('float32')


trainP, testP = trainModel(train, test)

print(trainP)
print("========================TEST")
print(testP)


#plot
forcast = np.concatenate((trainP,testP))
real = np.concatenate((valueTr[:,1],valueTs[:,1]))
plt.plot(real, color = 'blue', label = 'Serie Real')
plt.plot(forcast,"--", linewidth=2,color = 'red', label = 'Predicted Erlangs')
#plt.title('Erlangs Prediction--'+data_set.columns[choice])
plt.xlabel('Time')
plt.ylabel('Erlangs')
plt.legend()
plt.show()

#error
print("MSE: ", mse(real,forcast), " R2: ", r2_score(real,forcast))
#print(best_model.best_params_)

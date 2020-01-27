import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from math import sqrt

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

def steps_back(data):
    steps = data.shape[1] - 1
    values = data.values

    train = values[(steps):,:]

    x_test = train[:,:-1]
    y_test = train[:,-1]

    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]))
    
    print("x_test: %s" % str(x_test.shape))
    print("y_test: %s" % str(y_test.shape))

    return x_test, y_test

def split_timeSeries(data, percentage):
    length = data.shape[0]
    ntime = round(length * (percentage / 100))
    values = data.values

    train = values[:ntime, :]
    test = values[ntime:, :]

    print("train: %s" % str(train.shape))
    print("test: %s" % str(test.shape))

    return train, test

def trainModel(train_x, train_y):
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
    
    #prediction
    predicted_tr = model.predict(scaledTrain)

    # inverse_transform because prediction is done on scaled inputs
    predicted_tr = scalerOut.inverse_transform(predicted_tr.reshape(-1,1))

    return predicted_tr

def plotting(real, dataTest, prediction):
    values = real.values
    valuesTest = pd.DataFrame(dataTest)
    valuesTest = valuesTest.values
    forecast = np.concatenate((dataTest, prediction))
    plt.plot(real, color = 'blue', label = 'Serie Temporal de Potencia')
    plt.xlabel('Tiempo')
    plt.ylabel('Potencia')
    plt.title('Serie Temporal Muestreada cada 12 Hrs')
    plt.show()
    plt.plot(values, color = 'blue', label = 'Serie Original')
    #plt.plot([i for i in values] + [x for x in prediction])
    plt.plot(forecast, "-.", linewidth = 2, color = 'red', label = 'Prediccion')
    plt.plot(valuesTest, color = 'black', label = 'Entrenamiento')
    plt.xlabel('Tiempo')
    plt.ylabel('Potencia')
    plt.legend()
    plt.show()

def metrics(expected, predicted):
    print("****************************Metics :D************************************")
    plt.plot(expected, color = 'red', label = 'Serie Esperada')
    plt.plot(predicted, color = 'blue', label = 'Serie Predicha')
    
    #values = expected.values
    #expected = values[:,-1]
    mae = mean_absolute_error(expected, predicted)
    print("MAE: %f" % mae)
    mse = mean_squared_error(expected, predicted)
    print("MSE: %f" % mse)
    rmse = sqrt(mse)
    print("RMSE: %f" % rmse)
    plt.title("MAE: '%f', MSE: '%f', RMSE: '%f'" % (mae, mse, rmse))
    plt.legend()
    plt.show()
    return mae, mse, rmse

def addNewValue(x_test, newValue):
    for i in range(x_test.shape[1]-1):
        x_test[0][i] = x_test[0][i+1]

    x_test[0][x_test.shape[1]-1]=newValue

    return x_test

def predictFutureSeries(x_test, x_train):
    results = []
    for i in range(WINDOW):
        print("##################CICLE: %d" % i)
        parcial = trainModel(x_test, x_train)
        results.append(parcial[0])
        x_test = addNewValue(x_test,parcial[0])

    return results


WINDOW = 5
#Toma rangos de la serie temporal
print(df.head(n=10))
resampled = resampleSeries(df, 60)
print(resampled.head(n=10))
#Se divide la serie temporal en un porcentaje de muestras de entrenamiento
trainPart, testLength = split_timeSeries(resampled, 80)
WINDOW = testLength.shape[0]
#Se convierte la serie temporal a un problema de aprendizaje supervisado
reframed = series_to_supervised(trainPart, round(WINDOW), 1)
print(reframed.head(n=10))
#Se toman los pasos anteriores a entrenar
stepsBack, trainn = steps_back(reframed)
#Se procede a entrenar el algoritmo como los pasos anteriores y la x cantidad de ciclos
results = predictFutureSeries(stepsBack, trainn)
#Se convierte a dataFrame el array obtenido
adimen = [x for x in results]
prediction = pd.DataFrame(adimen)
prediction.columns = ['pronostico']
#Errores de la Serie Temporal
mae, mse, rmse = metrics(testLength, prediction)
#Se imprimen los resultados
plotting(resampled, trainPart, prediction)
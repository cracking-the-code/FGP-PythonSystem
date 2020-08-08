from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from math import sqrt
import pandas as pd
import numpy as np

def readCsv(path):
    print("Se procede a leer el archivo: " + path)
    df = pd.read_csv(path,  parse_dates=[0], header=0,index_col=0, squeeze=True)
    return df

def resampleSeries(data, resampleTime):
    print("Se procede a reajustar la serie temporal en: %d minuts" % resampleTime)
    dt = pd.DataFrame(data)
    if(resampleTime < 0.99):
        period = "%dS" % (resampleTime * 100)
        return dt.resample(period).mean()
    else:
        period = "%dT" % resampleTime
        return dt.resample(period).mean()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    print("Conversion a problema supervisado")
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

def steps_back(data):
    steps = data.shape[1] - 1
    values = data.values
    print("Se toman %d pasos anteriores de la serie temporal " % steps)
    train = values[(steps):,:]
    x_test = train[:,:-1]
    y_test = train[:,-1]
    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]))
    
    return x_test, y_test

def split_timeSeries(data, percentage):
    
    length = data.shape[0]
    ntime = round(length * (percentage / 100))
    values = data.values
    train = values[:ntime, :]
    test = values[ntime:, :]

    return train, test

def trainModel(train_x, train_y):
    print("Se procede a realizar el entrenamiento con el modelo SVR")
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

    print("Se procede a realizar la prediccion con el modelo SVR")
    predicted_tr = model.predict(scaledTrain)
    predicted_tr = scalerOut.inverse_transform(predicted_tr.reshape(-1,1))

    return predicted_tr

def plotting(real, dataTest, prediction):
    values = real.values
    valuesTest = pd.DataFrame(dataTest)
    valuesTest = valuesTest.values
    forecast = np.concatenate((dataTest, prediction))
    plt.plot(real, color = 'red', label = 'Serie Temporal de Potencia')
    plt.xlabel('Tiempo')
    plt.ylabel('Potencia')
    plt.show()
    plt.plot(forecast, "--", linewidth = 2, color = 'red', label = 'Prediccion')
    plt.plot(valuesTest, color = 'blue', label = 'Entrenamiento')
    plt.xlabel('Tiempo')
    plt.ylabel('Potencia')
    plt.legend()
    plt.show()

def metrics(expected, predicted):
    mae = mean_absolute_error(expected, predicted)
    mse = mean_squared_error(expected, predicted)
    rmse = sqrt(mse)

    return mae, mse, rmse

def addNewValue(x_test, newValue):
    for i in range(x_test.shape[1]-1):
        x_test[0][i] = x_test[0][i+1]

    x_test[0][x_test.shape[1]-1]=newValue

    return x_test

def predictFutureSeries(x_test, x_train):
    results = []
    for i in range(WINDOW):
        print("CICLO: %d" % i)
        parcial = trainModel(x_test, x_train)
        results.append(parcial[0])
        x_test = addNewValue(x_test,parcial[0])

    return results


WINDOW = 5
#Se lee el archivo CSV
csv = readCsv("january2008.csv")
print('CSV')
print(csv)
#Toma rangos de la serie temporal
resampled = resampleSeries(csv, 1440)
print('RESAMPLED')
print(resampled)
#Se divide la serie temporal en un porcentaje de muestras de entrenamiento
print('SPLIT')
trainPart, testLength = split_timeSeries(resampled, 90)
print('TRAIN')
print(trainPart)
print('TEST')
print(testLength)
WINDOW = testLength.shape[0]
print('WINDOW')
print(WINDOW)
#Se convierte la serie temporal a un problema de aprendizaje supervisado
reframed = series_to_supervised(trainPart, round(WINDOW), 1)
print('REFRAMED')
print(reframed)
#Se toman los pasos anteriores a entrenar
stepsBack, trainn = steps_back(reframed)
print('StepsBack')
print(stepsBack)
print('TRAINN')
print(trainn)
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
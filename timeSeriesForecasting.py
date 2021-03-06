import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

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

    print("=======================***********TRAIN RESULT***********=====================")
    print(predicted_tr)

    return predicted_tr

def plotting(data, prediction):
    values = data.values
    forecast = np.concatenate((data, prediction))
    plt.plot(values, color = 'blue', label = 'Serie Original')
    plt.plot(forecast, "--", linewidth = 2, color = 'red', label = 'Prediccion')
    plt.plot(prediction, color = 'green', label = 'prediction')
    plt.xlabel('Time')
    plt.ylabel('Potency')
    plt.legend()
    plt.show()

def addNewValue(x_test, newValue):
    for i in range(x_test.shape[1]-1):
        x_test[0][i] = x_test[0][i+1]

    x_test[0][x_test.shape[1]-1]=newValue

    return x_test

#Toma rangos de la serie temporal
resampled = resampleSeries(df, 30)
#Se convierte la serie temporal a un problema de aprendizaje supervisado
reframed = series_to_supervised(resampled, 5, 1)
#Se divide la serie en muestras de entrenamiento y testing
#x_train, y_train, x_test, y_test = split_timeSeries(reframed, 80)
stepsBack, trainn = steps_back(reframed)
#Se entrena al modelo y se obtienen predicciones de entrenamiento y testing
############trainP, testP = trainModel(x_train, y_train, x_test, y_test)
#Mostrar los resultados
#################################################################3plotting(resampled, trainP, testP)
#Agregar nuevos valores a la lista

x_test = stepsBack
print(stepsBack)

results=[]
for i in range(20):
    print("##################CICLE: %d" % i)
    parcial = trainModel(x_test, trainn)
    results.append(parcial[0])
    x_test = addNewValue(x_test,parcial[0])
    print("_____________________TEST_________________________")
    print(x_test)

adimen = [x for x in results]
prediction = pd.DataFrame(adimen)
prediction.columns = ['pronostico']

print(prediction)

plotting(resampled, prediction)
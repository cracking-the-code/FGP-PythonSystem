import pandas as pd
import matplotlib as plt

df = pd.read_csv('2020.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

df[['Voltage']].plot()
plt.pyplot.show()
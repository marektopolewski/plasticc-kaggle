import pandas as pd
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

color_map = {
    0 : 'b',
    1 : 'g',
    2 : 'r',
    3 : 'c',
    4 : 'm',
    5 : 'y'
}

metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')

series['flux'] = StandardScaler().fit_transform(series.flux)
series['mjd'] = StandardScaler().fit_transform(series.mjd)

x_cols = ['mjd','flux']
x = series[series.object_id==615]

y_same = series[series.object_id==26161]      # both objects of class 92
y_diff = series[series.object_id==9184]       # object of different class
y = y_same

for passb in range(0,6):

    x_p = x[x.passband==passb]
    y_p = y[y.passband==passb]

    distance, path = fastdtw(x_p, y_p, dist=euclidean)
    print(distance)

    plt.plot(x_p.mjd, x_p.flux, c = 'm')
    plt.plot(y_p.mjd, y_p.flux, c = 'y')
    plt.show()


# plt.scatter(x.mjd, x.flux, c=[color_map[p] for p in x.passband])
# plt.scatter(y.mjd, y.flux, c=[color_map[p] for p in y.passband])
# plt.show()

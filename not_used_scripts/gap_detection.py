import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def rateOfChange(flux_arr):
    change_arr = []
    flux_prev = flux_arr[0]
    for flux in flux_arr[1:]:
        change = flux - flux_prev
        change_arr.append(change)
    return change_arr

def getPhases(time_arr):
    diffs = [x - time_arr[i-1] for i, x in enumerate(time_arr)][1:]
    mean = int(np.mean(diffs))
    mean = 50
    df = pd.DataFrame(diffs, columns=['diff'])
    a = df[df['diff'] > mean]
    return len(a)


data = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
data = data.merge(metadata)

print(data[['object_id', 'target']].sample(5))

for obj in data.object_id.unique():
    obj_data = data[data.object_id==obj][['flux','mjd','passband']]

    print(getPhases(list(obj_data.mjd)))
    for passb in range(0,6):
        flux = obj_data[obj_data.passband==passb].flux
        mjd = obj_data[obj_data.passband==passb].mjd
        print(rateOfChange(list(flux)))
        plt.scatter(mjd, flux)
        plt.show()

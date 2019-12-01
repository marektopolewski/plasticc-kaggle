import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.svm import OneClassSVM

def get_aggregations():
    return {
        'passband': ['mean', 'std', 'var'],
        'flux': ['min', 'max', 'mean', 'median', 'std'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std'],
        'flux_ratio_sq' : ['sum'],
        'flux_by_flux_ratio_sq' : ['sum'],
        'detected': ['mean']
    }

def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

gc.enable()

metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
data = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')

data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']

x = data.groupby('object_id').agg(get_aggregations())
x.columns = get_new_columns(get_aggregations())

x['flux_diff'] = x['flux_max'] - x['flux_min']
x['flux_dif2'] = (x['flux_max'] - x['flux_min']) / x['flux_mean']
x['flux_w_mean'] = x['flux_by_flux_ratio_sq_sum'] / x['flux_ratio_sq_sum']
x['flux_dif3'] = (x['flux_max'] - x['flux_min']) / x['flux_w_mean']

x = x.reset_index().merge(right = metadata, how = 'outer', on = 'object_id')
mean_per_feature = x.mean(axis=0)
x.fillna(mean_per_feature, inplace=True)

del metadata
gc.collect()

x_out = x[x.target==92].sample(10)
x = x[x.target==42]

x = x.drop(['object_id', 'hostgal_photoz', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)
x_out = x_out.drop(['object_id', 'hostgal_photoz', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

x_test, x_train = x[0:10], x[10:]
x_test = x_test.append(x_out[0:2])
x_train = x_train.append(x_out[2:])

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(x_train)

out_train = clf.predict(x_train)
out_test = clf.predict(x_test)

print('Train:')
print(len(x_train),len(out_train[out_train==-1]))
print('\nTest:')
print(len(x_test),len(out_test[out_test==-1]))

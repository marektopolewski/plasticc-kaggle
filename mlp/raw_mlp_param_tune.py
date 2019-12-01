from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import gc

def get_aggregations():
    return {
        'passband': ['mean'],
        'flux': ['mean'],
        'flux_err': ['mean'],
        'flux_ratio_sq' : ['mean'],
        'flux_by_flux_ratio_sq' : ['mean'],
        'detected': ['mean']
    }

def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

gc.enable()

abs_path = '/modules/cs342/Assignment2/' 
metadata = pd.read_csv(abs_path + 'training_set_metadata.csv')
data = pd.read_csv(abs_path + 'training_set.csv')

data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']

series = data.groupby('object_id').agg(get_aggregations())
series.columns = get_new_columns(get_aggregations())

x = series.reset_index().merge(right = metadata, how = 'outer', on = 'object_id')
x.fillna(0, inplace=True)

del series, metadata
gc.collect()

t = x['target']
x = x.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

mlp = MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100), (100,100), (100,), (200,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [10**e for e in range(-6,1)],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=10, cv=3)
clf.fit(x,t)

print('\n\nBest parameters found:\n', clf.best_params_)

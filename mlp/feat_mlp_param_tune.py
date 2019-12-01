from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import gc

# Disable and enable printing (reduces messages from LombScargleMultibandFast)
import sys, os
import warnings
warnings.filterwarnings("ignore")

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def get_aggregations():
    return {
        'mjd': ['min', 'max', 'size'],
        'passband': ['min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
        'detected': ['mean'],
        'flux_ratio_sq':['sum','skew'],
        'flux_by_flux_ratio_sq':['sum','skew'],
    }

def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

def run():
    gc.enable()

    abs_path = '/modules/cs342/Assignment2/' 
    metadata = pd.read_csv(abs_path + 'training_set_metadata.csv')
    data = pd.read_csv(abs_path + 'training_set.csv')

    # add augumented data
    metadata_aug = pd.read_csv('../data_augmentation/aug_set_metadata.csv')
    data_aug = pd.read_csv('../data_augmentation/aug_set.csv')
    metadata = metadata.append(metadata_aug, sort=True)
    data = data.append(data_aug, sort=True)

    del metadata_aug, data_aug
    gc.collect()

    # aggregate columns
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

    x_inter = x[x.hostgal_photoz==0]
    x_extra = x[x.hostgal_photoz!=0]
    del x, metadata
    gc.collect()

    # define the parameter space to search within
    parameter_space = {
        'hidden_layer_sizes': [(50,100,50), (100,100,100), (100,100), (100,), (200,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [10**e for e in range(-7,-2)],
        'learning_rate': ['constant','adaptive'],
    }

    # inter galactic objects, calculate the period
    t_inter = x_inter['target']
    periods_data = pd.read_csv('../feature_extraction/training_periods.csv')
    x_inter = x_inter.merge(periods_data.drop('period_score', axis=1), on='object_id')
    x_inter = x_inter.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    x_inter = StandardScaler().fit_transform(x_inter)

    blockPrint()
    mlp = MLPClassifier(max_iter=100)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_inter,t_inter)
    enablePrint()
    print('\n\nBest parameters galactic found:\n', clf.best_params_)


    # extra galactic objects, calculate the absolute magnitude
    t_extra = x_extra['target']
    abs_magn_data = pd.read_csv('../feature_extraction/training_abs_magnitudes.csv')
    x_extra = x_extra.merge(abs_magn_data, on='object_id')
    x_extra = x_extra.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    x_extra = StandardScaler().fit_transform(x_extra)

    blockPrint()
    mlp = MLPClassifier(max_iter=100)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(x_extra,t_extra)
    enablePrint()
    print('\n\nBest parameters extragalactic found:\n', clf.best_params_)

run()
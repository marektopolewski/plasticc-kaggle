import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

from scipy.signal import lombscargle
from tools import get_objects_by_id as getById

# feature extraction methods
def getLombScale(time, flux):
    f = np.arange(0.5,30,0.5)
    freqs = lombscargle(time.values, flux.values, f)
    max_freqs = np.argmax(freqs[1:]) + 1
    return max_freqs

def test_makePeriod(obj_id, data, metadata):
    flux = data[data.object_id==obj_id].flux
    time = data[data.object_id==obj_id].mjd
    period = getLombScale(flux, time)
    return period

def test_makeAbsMagn(obj_id, data, metadata):
    flux_det = data[(data.object_id==obj_id) & (data.detected == 1)].flux
    flux = data[(data.object_id==obj_id)].flux
    distmod = list(metadata[metadata.object_id == obj_id].distmod)[0]

    true_flux = np.max(flux_det) - np.median(flux)
    if true_flux < 1: true_flux = 1

    magnitude = -2.5 * np.log(true_flux)
    abs_magn = magnitude - distmod
    return abs_magn

def makeAbsMagn(data, metadata):
    abs_magnitudes = []
    for obj_id in metadata.object_id:
        flux_det = data[(data.object_id==obj_id) & (data.detected == 1)].flux
        flux = data[(data.object_id==obj_id)].flux
        distmod = list(metadata[metadata.object_id == obj_id].distmod)[0]

        true_flux = np.max(flux_det) - np.median(flux)
        if true_flux < 1: true_flux = 1

        magnitude = -2.5 * np.log(true_flux)
        abs_magn = magnitude - distmod

        abs_magnitudes.append(abs_magn)
    return abs_magnitudes

# aggregation methods
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

def get_class_labels(classes):
    return ['class_'+str(c) for c in classes]

# predict probabilities for extra and intra galactic
def predict():

    # feature extraction
    gc.enable()
    metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
    data = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
    print('[TRAIN] data loaded.')

    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']

    series = data.groupby('object_id').agg(get_aggregations())
    series.columns = get_new_columns(get_aggregations())

    series['flux_diff'] = series['flux_max'] - series['flux_min']
    series['flux_dif2'] = (series['flux_max'] - series['flux_min']) / series['flux_mean']
    series['flux_w_mean'] = series['flux_by_flux_ratio_sq_sum'] / series['flux_ratio_sq_sum']
    series['flux_dif3'] = (series['flux_max'] - series['flux_min']) / series['flux_w_mean']

    x = series.reset_index().merge(right = metadata, how = 'outer', on = 'object_id')
    # mean_per_feature = x.mean(axis=0)
    x.fillna(0, inplace=True)

    del series, metadata
    gc.collect()

    x_inter = x[x.hostgal_photoz==0]
    x_extra = x[x.hostgal_photoz!=0]
    del x
    gc.collect()
    print('[TRAIN] features extracted.')

    # train inter galactic objects
    t_inter = x_inter['target']
    periods_data = pd.read_csv('../feature_extraction/training_periods.csv')
    x_inter = x_inter.merge(periods_data.drop('period_score', axis=1), on='object_id')
    x_inter = x_inter.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    inter_model = RandomForestClassifier(n_estimators=100)
    inter_model.fit(x_inter, t_inter)

    del x_inter, t_inter
    gc.collect()
    print('[TRAIN] inter-galactic model trained.')

    # train extra galactic objects
    t_extra = x_extra['target']
    abs_magns = pd.read_csv('../feature_extraction/training_abs_magnitudes.csv')
    x_extra = x_extra.merge(abs_magns, on='object_id')
    x_extra = x_extra.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    extra_model = RandomForestClassifier(n_estimators=100)
    extra_model.fit(x_extra, t_extra)

    del x_extra, t_extra, data
    gc.collect()
    print('[TRAIN] extra-galactic model trained.')


    # load test data
    metadata = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
    print('[TEST] metadata loaded.')

    # https://www.kaggle.com/kyleboone/naive-benchmark-galactic-vs-extragalactic
    inter_classes = list(inter_model.classes_)
    extra_classes = list(extra_model.classes_)
    class_labels = get_class_labels(inter_classes + extra_classes)

    filepath = '/modules/cs342/Assignment2/test_set.csv'
    counter = 0

    for object_id, df in getById(filepath):

        counter += 1
        index = metadata[metadata.object_id==object_id].index.values[0]

        df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
        df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

        series = df.groupby('object_id').agg(get_aggregations())
        series.columns = get_new_columns(get_aggregations())

        series['flux_diff'] = series['flux_max'] - series['flux_min']
        series['flux_dif2'] = (series['flux_max'] - series['flux_min']) / series['flux_mean']
        series['flux_w_mean'] = series['flux_by_flux_ratio_sq_sum'] / series['flux_ratio_sq_sum']
        series['flux_dif3'] = (series['flux_max'] - series['flux_min']) / series['flux_w_mean']

        x = series.reset_index().merge(metadata)
        x.fillna(0, inplace=True)

        del series
        gc.collect()

        preds_df = pd.DataFrame(columns = inter_classes + extra_classes)
        if metadata.hostgal_photoz[index] == 0:
            x['period'] = test_makePeriod(object_id, df, x)
            x = x.drop(['object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

            pred = inter_model.predict_proba(x)
            pred = pd.DataFrame(pred, columns = inter_classes)
            for extra_class in extra_classes:
                pred[extra_class] = 0

        else:
            x['abs_magn'] = test_makeAbsMagn(object_id, df, x)
            x = x.drop(['object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)
            pred = extra_model.predict_proba(x)
            pred = pd.DataFrame(pred, columns = extra_classes)
            for inter_class in inter_classes:
                pred[inter_class] = 0

        preds_df = preds_df.append(pred, sort=True)
        preds_df.columns = class_labels
        preds_df['class_99'] = 0
        preds_df['object_id'] = object_id

        if counter==1 :
            preds_df.to_csv('preds_gal.csv', header=True, mode='a', index=False)
            first = False
        else:
            preds_df.to_csv('preds_gal.csv', header=False, mode='a', index=False)

        del preds_df
        gc.collect()

        if counter % (len(metadata)/20) == 0 :
            progress = 100 * float(counter) / len(metadata)
            print(' [TEST] progress: %.2f prec.' % progress)


    print('[END] Completed.')

predict()

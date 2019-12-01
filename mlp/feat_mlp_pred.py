import pandas as pd
import numpy as np
import gc

from sklearn.neural_network import MLPClassifier

# aggregation methods
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

def get_class_labels(classes):
    return ['class_'+str(c) for c in classes]

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

# predict probabilities for extra and intra galactic
def predict():

    # feature extraction
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

    # inter galactic objects, calculate the period
    t_inter = x_inter['target']
    periods_data = pd.read_csv('../feature_extraction/training_periods.csv')
    x_inter = x_inter.merge(periods_data.drop('period_score', axis=1), on='object_id')
    x_inter = x_inter.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    # model = MLPClassifier()
    model = MLPClassifier(
        alpha = 10**(-6),
        activation = 'tanh',
        solver = 'adam',
        learning_rate = 'adaptive',
        hidden_layer_sizes = (100,100)
    )
    inter_model.fit(x_inter, t_inter)

    del x_inter, t_inter
    gc.collect()
    print('[TRAIN] galactic model trained.')

    # train extra galactic objects
    t_extra = x_extra['target']
    abs_magns = pd.read_csv('../feature_extraction/training_abs_magnitudes.csv')
    x_extra = x_extra.merge(abs_magns, on='object_id')
    x_extra = x_extra.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    extra_model = MLPClassifier(
        alpha = 10**(-3),
        activation = 'relu',
        solver = 'adam',
        learning_rate = 'adaptive',
        hidden_layer_sizes = (100,100,100)  
    )
    extra_model.fit(x_extra, t_extra)

    del x_extra, t_extra, data
    gc.collect()
    print('[TRAIN] extra-galactic model trained.')

    # load test data
    metadata = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
    print('[TEST] metadata loaded.')

    class_labels = get_class_labels(model.classes_)

    chunks = 50000000
    for i_c, df in enumerate(pd.read_csv('/modules/cs342/Assignment2/test_set.csv', chunksize=chunks, iterator=True)):

        # aggregate columns
        df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
        df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

        x = df.groupby('object_id').agg(get_aggregations())
        x.columns = get_new_columns(get_aggregations())

        x['flux_diff'] = x['flux_max'] - x['flux_min']
        x['flux_dif2'] = (x['flux_max'] - x['flux_min']) / x['flux_mean']
        x['flux_w_mean'] = x['flux_by_flux_ratio_sq_sum'] / x['flux_ratio_sq_sum']
        x['flux_dif3'] = (x['flux_max'] - x['flux_min']) / x['flux_w_mean']

        x = x.reset_index().merge(right = metadata, how = 'outer', on = 'object_id')
        mean_per_feature = x.mean(axis=0)
        x.fillna(mean_per_feature, inplace=True)

        del series, df
        gc.collect()

        object_id = x['object_id']
        x = x.drop(['object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)
        pred = model.predict_proba(x)

        del x
        gc.collect()

        preds_df = pd.DataFrame(pred, columns = class_labels)

        preds_df['class_99'] = 0
        preds_df['object_id'] = object_id

        if i_c == 0:
            preds_df.to_csv('temp_pred.csv', header=True, mode='a', index=False)
        else:
            preds_df.to_csv('temp_pred.csv', header=False, mode='a', index=False)

        del preds_df
        gc.collect()

        if (i_c + 1) % 2 == 0:
            chunks_done = chunks * (i_c+1)
            print(' [TEST PROGRESS] %i done.' % chunks_done)

    z = pd.read_csv('temp_pred.csv')

    import os, errno
    try: os.remove('temp_pred.csv')
    except OSError: pass

    z = z.groupby('object_id').mean()
    z.to_csv('pred_raw_mlp.csv', index=True)

    print('[END] Completed.')

predict()

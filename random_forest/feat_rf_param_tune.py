import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from scipy.signal import lombscargle

def addClass99(preds, preds_99):
    outliers = len(preds_99[preds_99==-1])
    print('For all %i records, outliers=%i' % (len(preds),outliers))

# feature extraction methods
def getLombScale(time, flux):
    f = np.arange(0.5,30,0.5)
    freqs = lombscargle(time.values, flux.values, f)
    max_freqs = np.argmax(freqs[1:]) + 1
    return max_freqs

def make_periods(data, metadata):
    periods = []
    for obj_id in metadata.object_id:
        flux = data[data.object_id==obj_id].flux
        time = data[data.object_id==obj_id].mjd
        period_i = getLombScale(flux, time)
        periods.append(period_i)
    return periods

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

x = x.reset_index().merge(right=metadata, how='outer', on='object_id')
mean_per_feature = x.mean(axis=0)
x.fillna(mean_per_feature, inplace=True)

x_inter = x[x.hostgal_photoz == 0]
x_extra = x[x.hostgal_photoz != 0]
del x, metadata
gc.collect()

t_inter = x_inter['target']
x_inter['period'] = make_periods(data, x_inter)
x_inter = x_inter.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

t_extra = x_extra['target']
x_extra['abs_magn'] = makeAbsMagn(data, x_extra)
x_extra = x_extra.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

from scipy.stats import randint as sp_randint
estimators = [100,150,200,300]
param_dist = {"max_depth": [10, 20, 30, 40, 50, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "n_estimators": estimators}



print('\n ################# INTER-GALACTIC ################# \n')
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_dist,
    n_iter = 100,
    cv = 3,
    random_state=42,
    n_jobs = 10
)
rf_random.fit(x_inter, t_inter)
print('Best score: %.5f' % rf_random.best_score_)
print(rf_random.best_params_)

x_inter = StandardScaler().fit_transform(x_inter)
print('\n ############# SCALED INTER-GALACTIC ############## \n')
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_dist,
    n_iter = 100,
    cv = 3,
    random_state=42,
    n_jobs = 10
)
rf_random.fit(x_inter, t_inter)
print('Best score: %.5f' % rf_random.best_score_)
print(rf_random.best_params_)


print('\n ################# EXTRA-GALACTIC ################# \n')
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_dist,
    n_iter = 100,
    cv = 3,
    random_state=42,
    n_jobs = 10
)
rf_random.fit(x_extra, t_extra)
print('Best score: %.5f' % rf_random.best_score_)
print(rf_random.best_params_)

x_extra = StandardScaler().fit_transform(x_extra)
print('\n ############# SCALED EXTRA-GALACTIC ############## \n')
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator = rf,
    param_distributions = param_dist,
    n_iter = 100,
    cv = 3,
    random_state=42,
    n_jobs = 10
)
rf_random.fit(x_extra, t_extra)
print('Best score: %.5f' % rf_random.best_score_)
print(rf_random.best_params_)
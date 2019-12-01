import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

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

# ------------------------------------------------------------------------------------
# helper function for generating column names for an attribute and aggregation method
def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

# kfold for extragalactic and galactic separately
def score_kfold(splits):

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

    # perform k-fold CV for the galactic class
    kfold = KFold(n_splits=splits, random_state=42)
    score = 0
    for train_i,test_i in kfold.split(x_inter):
        x_train, x_test = x_inter.iloc[train_i], x_inter.iloc[test_i]
        t_train, t_test = t_inter.iloc[train_i], t_inter.iloc[test_i]

        x_train_period, x_test_period = x_train['period'], x_test['period']
        x_train, x_test = x_train.drop('period', axis=1), x_test.drop('period', axis=1)
        scale = StandardScaler()
        scale.fit(x_train)
        x_train = pd.DataFrame(scale.transform(x_train), columns=x_train.columns)
        x_test  = pd.DataFrame(scale.transform(x_test), columns=x_test.columns)
        x_train['period'] = list(x_train_period)
        x_test['period'] = list(x_test_period)

        # tuned RFC for galactic class using hyper-param tuning
        model = RandomForestClassifier(
            max_features = 5, 
            min_samples_split = 5, 
            bootstrap = False,
            n_estimators = 300, 
            max_depth = 10
        )
        model.fit(x_train, t_train)
        preds = model.predict_proba(x_test)

        score_i = log_loss(t_test, preds)
        print("[LOGLOSS-KFOLD] Fold result: %.3f" % score_i)
        score += score_i

    inter_score = float(score)/splits
    print('Score inter galactic = %.5f' % inter_score)


    # extra galactic objects, calculate the absolute magnitude
    t_extra = x_extra['target']
    abs_magn_data = pd.read_csv('../feature_extraction/training_abs_magnitudes.csv')
    x_extra = x_extra.merge(abs_magn_data, on='object_id')
    x_extra = x_extra.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    # perform k-fold CV for the extragalactic class
    kfold = KFold(n_splits=splits, shuffle=False)
    score = 0
    for train_i,test_i in kfold.split(x_extra):
        x_train, x_test = x_extra.iloc[train_i], x_extra.iloc[test_i]
        t_train, t_test = t_extra.iloc[train_i], t_extra.iloc[test_i]

        # it has been tested, and not scaling the extracted features yeilds better results
        x_train_abs_magn, x_test_abs_magn = x_train['abs_magn'], x_test['abs_magn']
        x_train, x_test = x_train.drop('abs_magn', axis=1), x_test.drop('abs_magn', axis=1)
        scale = StandardScaler()
        scale.fit(x_train)
        x_train = pd.DataFrame(scale.transform(x_train), columns=x_train.columns)
        x_test  = pd.DataFrame(scale.transform(x_test), columns=x_test.columns)
        x_train['abs_magn'] = list(x_train_abs_magn)
        x_test['abs_magn'] = list(x_test_abs_magn)

        # tuned RFC for extragalactic class using hyper-param tuning
        model = RandomForestClassifier(
            max_features = 9, 
            min_samples_split = 7, 
            bootstrap = True,
            n_estimators = 300, 
            max_depth = 10
        )
        model.fit(x_train, t_train)
        preds = model.predict_proba(x_test)

        score_i = log_loss(t_test, preds)
        print("[LOGLOSS-KFOLD] Fold result: %.3f" % score_i)
        score += score_i

    extra_score = float(score)/splits
    print('Score extra galactic = %.5f' % extra_score)

    score = (extra_score+inter_score)/2
    print('\nOverall score = %.5f' % score)


score_kfold(5)
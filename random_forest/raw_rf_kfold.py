import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

# aggregation methods
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

def get_class_labels(classes):
    return ['class_'+str(c) for c in classes]

# kfold for extra and intra galactic
def score_kfold(splits):

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

    kfold = KFold(n_splits=splits, random_state=42)
    score = 0
    for train_i,test_i in kfold.split(x):
        x_train, x_test = x.iloc[train_i], x.iloc[test_i]
        t_train, t_test = t.iloc[train_i], t.iloc[test_i]

        # model = RandomForestClassifier()
        model = RandomForestClassifier(
            max_features=5,
            min_samples_split=2,
            bootstrap=True,
            n_estimators=300,
            max_depth=10
        )
        # {'max_features': 5, 'min_samples_split': 2, 'bootstrap': True, 'n_estimators': 300, 'max_depth': 10}

        model.fit(x_train, t_train)
        preds = model.predict_proba(x_test)

        score_i = log_loss(t_test, preds)
        print("[LOGLOSS-KFOLD] Fold result: %.3f" % score_i)
        score += score_i

    score = float(score)/splits
    print('Score = %.5f' % score)


score = score_kfold(5)

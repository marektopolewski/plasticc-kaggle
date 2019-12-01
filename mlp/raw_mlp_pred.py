import pandas as pd
import numpy as np
import gc

from sklearn.neural_network import MLPClassifier

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

    x = series.reset_index().merge(right = metadata, how = 'outer', on = 'object_id')
    x.fillna(0, inplace=True)

    del series, metadata
    gc.collect()
    print('[TRAIN] features extracted.')

    t = x['target']
    x = x.drop(['target', 'object_id', 'hostgal_specz', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf'], axis=1)

    # model = MLPClassifier()
    model = MLPClassifier(
        alpha = 10**(-6),
        activation = 'tanh',
        solver = 'adam',
        learning_rate = 'adaptive',
        hidden_layer_sizes = (50, 100, 50)
    )
    model.fit(x, t)

    del x, t
    gc.collect()
    print('[TRAIN] model trained.')

    # load test data
    metadata = pd.read_csv('/modules/cs342/Assignment2/test_set_metadata.csv')
    print('[TEST] metadata loaded.')

    class_labels = get_class_labels(model.classes_)

    chunks = 50000000
    for i_c, df in enumerate(pd.read_csv('/modules/cs342/Assignment2/test_set.csv', chunksize=chunks, iterator=True)):

        df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
        df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

        series = df.groupby('object_id').agg(get_aggregations())
        series.columns = get_new_columns(get_aggregations())

        x = series.reset_index().merge(metadata)
        x.fillna(0, inplace=True)

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

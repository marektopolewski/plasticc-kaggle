from __future__ import division
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import tensorflow as tf
import keras.backend as K

# I could not import keras.utils.to_categorical, so I directly copied it from github
# from keras.utils import to_categorical
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/utils/np_utils.py
def to_categorical(y, num_classes=None):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=np.float32)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


def get_class_labels(classes):
    return ['class_'+str(c) for c in classes]

"""
CNN Kfold
https://towardsdatascience.com/the-softmax-function-neural-net-outputs-as-probabilities-and-ensemble-classifiers-9bd94d75932
https://faroit.github.io/keras-docs/1.2.2/
https://www.kaggle.com/higepon/updated-keras-cnn-use-time-series-data-as-is
"""
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense, MaxPooling1D, Lambda, Activation, Input, BatchNormalization
from keras.callbacks import EarlyStopping

batch_size = 256
def make_cnn():
    # initial inputs to the NN
    inputs = Input(shape=(X_train.shape[1], 6), dtype = 'float32', name = 'input0')

    # consecutive layers of the NN
    output = Convolution1D(
        nb_filter = batch_size, filter_length = 80,
        border_mode = 'same',
        init = 'glorot_uniform',
        W_regularizer = l2(l=0.0001)
    ) (inputs)
    output = BatchNormalization() (output)
    output = Activation('relu') (output)
    output = MaxPooling1D(
        pool_length = 4, stride = None
    ) (output)
    output = Convolution1D(
        nb_filter = batch_size, filter_length = 3,
        border_mode = 'same',
        init = 'glorot_uniform',
        W_regularizer = l2(l=0.0001)
    ) (output)
    output = BatchNormalization() (output)
    output = Activation('relu') (output)
    output = MaxPooling1D(
        pool_length = 4, stride = None
    ) (output)
    output = Lambda(lambda x: K.mean(x, axis=1)) (output)
    output = Dense(len(classes), activation = 'softmax') (output)

    # generate a keras NN model
    cnn = Model(input = inputs, output = output)
    return cnn

"""
Prepare data
https://www.kaggle.com/higepon/updated-keras-cnn-use-time-series-data-as-is
"""
print('Loading data...')

# load, augument, scale and sort series data
train = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
train_aug = pd.read_csv('../aug_data/aug_set.csv')
train = train.append(train_aug, sort=True)

test = pd.read_csv('/modules/cs342/Assignment2/test_set.csv')
test_obj = test.object_id

# load and augument metadata
meta_train = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
meta_aug = pd.read_csv('../aug_data/aug_set_metadata.csv')
meta_train = meta_train.append(meta_aug, sort=True)

meta_test = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')

print('Data loaded.')
print('Preparing observations...')

# scale data, NN are highly sensitive to normalisation
scaled_train = StandardScaler().fit_transform(train[['mjd', 'flux', 'flux_err']])
train[['mjd', 'flux', 'flux_err']] = scaled_train
train = train.sort_values(['object_id', 'passband', 'mjd'])

scaled_test = StandardScaler().fit_transform(test[['mjd', 'flux', 'flux_err']])
test[['mjd', 'flux', 'flux_err']] = scaled_test
test = test.sort_values(['object_id', 'passband', 'mjd'])

# transform features, so that all attributes (flux, flux error and detected flag)
# for a given object is contained within a single row
train = train.groupby(['object_id', 'passband'])['flux', 'flux_err', 'detected']
train = train.apply(lambda df: df.reset_index(drop=True))
train = train.unstack()
train.fillna(0, inplace=True)

test = test.groupby(['object_id', 'passband'])['flux', 'flux_err', 'detected']
test = test.apply(lambda df: df.reset_index(drop=True))
test = test.unstack()
test.fillna(0, inplace=True)

# reshape the data
X_train = train.values.reshape(-1, 6, len(train.columns)).transpose(0, 2, 1)
X_test = train.values.reshape(-1, 6, len(test.columns)).transpose(0, 2, 1)

# make dictionary to find an index of corresponding class in the target map
classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
class_map = dict()
counter = 0
for val in classes:
    class_map[val] = counter
    counter += 1

# Generate the output target map from
train_x_passb_0 = train.reset_index()
train_x_passb_0 = train_x_passb_0[train_x_passb_0.passband == 0]

# ensure uniqueness of x by selecting only one passband of the transfomed x matrix
merged_meta_train = train_x_passb_0.merge(meta_train, on='object_id', how='left')
merged_meta_train.fillna(0, inplace=True)

targets = merged_meta_train.target
target_map = np.zeros((targets.shape[0],))
target_map = np.array([class_map[val] for val in targets])
t_categorical = to_categorical(target_map)


# Predict
folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=1)
best_loss = 999
for fold_, (train_i, test_i) in enumerate(folds.split(target_map, target_map)):

    x_train, t_train = X_train[train_i], t_categorical[train_i]
    x_test , t_test  = X_train[test_i] , t_categorical[test_i]

    model = make_cnn()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    stopping = EarlyStopping(monitor='test_loss', patience=60, verbose=0, mode='auto')

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(
        x = x_train, y = t_train,
        validation_data = [x_test, t_test],
        nb_epoch = 100, batch_size = batch_size,
        shuffle = False, verbose = 1
    )

    t_pred = model.predict(x_test, batch_size = batch_size)
    loss = log_loss(t_test, t_pred)

    print('\nModel trained, score: %.4f\n' % loss)
    if loss < best_loss: best_model = model

print('\Training finished. Predicting: ')

preds = model.predict(X_test, batch_size = batch_size)
preds = pd.DataFrame(preds, columns=get_class_labels(classes))
preds['object_id'] = list(test_obj)

preds.to_csv('cnn_predictions.csv', header=True, index=False)
print('Predictions exported.')

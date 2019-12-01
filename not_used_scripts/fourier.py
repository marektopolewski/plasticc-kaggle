import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import chain

# ------------------------------------------------------------------------------------------------



train_metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
train_series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')

groups = train_series.groupby(['object_id', 'passband'])

def normalise(ts):
    return (ts - ts.mean()) / ts.std()

# times of recording each passband for every object
times = groups.apply(
    lambda block: block['mjd'].values
)
times = times.reset_index().rename(columns={0: 'seq'})

# normalise each flux observation per object
flux = groups.apply(
    lambda block: normalise(block['flux']).values
)
flux = flux.reset_index().rename(columns={0: 'seq'})
print(flux)

times_list = times.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()
flux_list = flux.groupby('object_id').apply(lambda x: x['seq'].tolist()).tolist()

def extract_freq(t, m, e):
    fs = np.linspace(2*np.pi/0.1, 2*np.pi/500, 10000)
    pgram = signal.lombscargle(t, m, fs, normalize=True)
    return fs[np.argmax(pgram)]

import cesium.featurize as featurize
from scipy import signal
N = 100
cfeats = featurize.featurize_time_series(times=times_list[:N],
                                        values=flux_list[:N],
                                        features_to_use=['freq1_freq',
                                                        'freq1_signif',
                                                        'freq1_amplitude1'],
                                        scheduler=None)
cfeats.stack('channel').iloc[:24]
def plot_phase(n, fr):
    selected_times = times_list[n]
    selected_flux = flux_list[n]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    f, ax = plt.subplots(figsize=(12, 6))
    for band in range(6):
        ax.scatter(x=(selected_times[band] * fr) % 1,
                   y=selected_flux[band],
                   c=colors[band])
    ax.set_xlabel('phase')
    ax.set_ylabel('relative flux')
    ax.set_title('object '+train_metadata["object_id"][n]+', class '+train_metadata["target"][n])
    plt.show()

plot_phase(0, 3.081631)


# 1
# train_metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
# train_series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
#
# model = LombScargleMultiband(fit_period=True)
# group = train_series[train_series.object_id==615]
#
# print(group)
#
# t, f, e, b = group['mjd'], group['flux'], group['flux_err'], group['passband']
# model.optimizer.period_range = (0.1, int((group['mjd'].max() - group['mjd'].min()) / 2))
# model.fit(t, f, e, b)
#
# print(model.best_period)
# tfit = np.linspace(0, model.best_period, 1000)
# magfit = model.predict(tfit, filts='g')

# 2
# train_metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
# train_series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')
#
# def time_kernel(diff, tau):
#     return np.exp(-diff ** 2 / (2 * tau ** 2))
#
# def group_transform(chunk):
#     sample_weights = chunk[['sw_'+str(i) for i in range(len(sample_points))]]
#     sample_weights /= np.sum(sample_weights, axis=0)
#     weighted_flux = np.expand_dims(chunk['flux'].values, 1) * sample_weights.fillna(0)
#     return np.sum(weighted_flux, axis=0)
#
# t_min, t_max = train_series['mjd'].min(), train_series['mjd'].max()
#
# sample_points = np.array(np.arange(t_min, t_max, 20))
#
# weights = time_kernel(np.expand_dims(sample_points, 0) - np.expand_dims(train_series['mjd'].values, 1), 5)
# ts_mod = train_series[['object_id', 'mjd', 'passband', 'flux']].copy()
# for i in range(len(sample_points)):
#     ts_mod['sw_'+str(i)] = weights[:, i]
#
# ts_samp = ts_mod[ts_mod['object_id'].isin([615, 713])].groupby(['object_id', 'passband']).apply(group_transform)
#
# print(ts_samp)

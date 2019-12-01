import pandas as pd
import numpy as np
import gc

from scipy.signal import lombscargle
from gatspy.periodic import LombScargleMultibandFast

# -----------------------------------------------------------------------------------
# Disable and enable printing (reduces messages from LombScargleMultibandFast)
import sys, os
import warnings
warnings.filterwarnings("ignore")

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# ------------------------------------------------------------------------------------
# feature extraction methods

# calculate period using LombScargle from scipy.signal
def getPeriod_1(data):
    time = data.mjd
    flux = data.flux

    f = np.arange(0.5,30,0.5)
    freqs = lombscargle(time.values, flux.values, f)
    max_freqs = np.argmax(freqs[1:]) + 1
    return max_freqs

# calculate period using LombScargle Mulitband Fast from gatspy.periodic
def getPeriod(data):
    blockPrint()
    t, f, e, b = data.mjd, data.flux, data.flux_err, data.passband
    lsm = LombScargleMultibandFast(fit_period=True)
    period_range = (0.1, int((data.mjd.max() - data.mjd.min()) / 2))
    lsm.optimizer.period_range = period_range
    lsm.fit(t, f, e, b)
    enablePrint()
    return lsm.best_period, lsm.score(lsm.best_period)

# generate period for each sample in the given data frames
def extract_periods(data, metadata, filename):
    print('\nCalculating periods for galactic data')
    count, max_count = 0, len(metadata.object_id)

    periods, scores = [], []
    for obj_id in metadata.object_id:
        count += 1
        obj_data = data[data.object_id==obj_id]

        period_i, score_i = getPeriod(obj_data)
        periods.append(period_i)
        scores.append(score_i)

        if int(count % (max_count/5)) == 0:
            progress = int(count*100.0/max_count)
            print('> period calc progress: %i perc.' % progress)
    
    df_periods = pd.DataFrame(list(metadata.object_id), columns=['object_id'])
    df_periods['period'] = periods
    df_periods['period_score'] = scores

    filename_period = filename + 'periods.csv'
    df_periods.to_csv(filename_period, header=True, index=False)
    print('Periods caluclated and exported')

# generate absolute magnitude (formula in the report)
def extract_abs_magn(data, metadata, filename):
    print('\nCalculating magnitudes for extragalactic data')
    count, max_count = 0, len(metadata.object_id)

    abs_magnitudes = []
    flux_median = np.median(data.flux)
    
    for obj_id in metadata.object_id:
        count += 1
        flux_det = data[(data.object_id==obj_id) & (data.detected == 1)].flux
        flux = data[(data.object_id==obj_id)].flux
        distmod = list(metadata[metadata.object_id == obj_id].distmod)[0]

        true_flux = np.max(flux_det) - flux_median
        if true_flux < 1: true_flux = 1

        magnitude = -2.5 * np.log(true_flux)
        abs_magn = magnitude - distmod
        abs_magnitudes.append(abs_magn)

        if int(count % (max_count/5)) == 0:
            progress = int(count*100.0/max_count)
            print('> abs magnitudes calc progress: %i perc.' % progress)

    df_abs_magn = pd.DataFrame(list(metadata.object_id), columns=['object_id'])
    df_abs_magn['abs_magn'] = abs_magnitudes

    filename_abs_magn = filename + 'abs_magnitudes.csv'
    df_abs_magn.to_csv(filename_abs_magn, header=True, index=False)
    print('Magnitudes caluclated and exported')

# feature extraction and saving to CSV files
def extract(test_data_flag):

    gc.enable()

    abs_path, path = '/modules/cs342/Assignment2/', 'test_'
    if not test_data_flag: path = 'training_'

    print('Loading data...')
    metadata = pd.read_csv(abs_path + path + 'set_metadata.csv')
    data = pd.read_csv(abs_path + path + 'set.csv')

    # add augumented data if generating for training data
    if not test_data_flag:
        aug_path = '../data_augmentation/'
        metadata_aug = pd.read_csv(aug_path + 'aug_set_metadata.csv')
        data_aug = pd.read_csv(aug_path + 'aug_set.csv')
        metadata = metadata.append(metadata_aug)
        data = data.append(data_aug)

        del metadata_aug, data_aug
        gc.collect()


    # extract absolute magnitudes
    metadata_extragalactic = metadata[metadata.hostgal_photoz!=0]
    extract_abs_magn(data, metadata_extragalactic[0:5], path)

    del metadata_extragalactic
    gc.collect()
    

    # extract periods
    metadata_galactic = metadata[metadata.hostgal_photoz==0]
    extract_periods(data, metadata_galactic, path)

    del metadata_galactic
    gc.collect()


extract(True)
"""
Data augumentation of the training set
Inspired by https://www.kaggle.com/mithrillion/all-classes-light-curve-characteristics-updated
"""
import pandas as pd
import numpy as np

from gatspy.periodic import LombScargleMultibandFast 
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------------------------------------------------
# Disable and enable printing (reduces messages from LombScargleMultibandFast)
import sys, os
import warnings
warnings.filterwarnings("ignore")

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# -----------------------------------------------------------------------------------
# laod training data
metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv')
series = pd.read_csv('/modules/cs342/Assignment2/training_set.csv')

# initialise augumatend data frames
new_metadata = pd.DataFrame(columns=metadata.columns)
new_series = pd.DataFrame(columns=series.columns)

# last object id in the train set
new_id = int(series.object_id.max())

print('[0/2] Starting data augmentation...\n')

# -----------------------------------------------------------------------------------
# augument with this method only galactic data
galactic_objects = metadata[metadata.hostgal_photoz==0]
reproduce = galactic_objects.object_id


# Uncomment to generate fewer galactic samples, classification performs well already
# galactic_classes = galactic_objects.target.unique()
# reproduce = [
#     galactic_objects[galactic_objects.target==g_c].sample(5).object_id
#     for g_c in galactic_classes
# ]
# reproduce = list(np.array(reproduce).flat)

blockPrint()
count, max_count = 0, len(reproduce)
for obj_id in reproduce:

    count += 1
    new_id = new_id + 1
    subset = series[series.object_id==obj_id]

    # train model to estimate the underlying function of flux for each passband
    # and optimise to find the best period esitmiation
    t, f, e, b = subset.mjd, subset.flux, subset.flux_err, subset.passband
    lsm = LombScargleMultibandFast(fit_period=True)
    date_min, date_max = subset.mjd.max(), subset.mjd.min()
    period_range = (0.1, int((date_min - date_max)/2))
    lsm.optimizer.period_range = period_range
    lsm.fit(t, f, e, b)

    # find best period and corresponding time period to shift the data samples
    period = lsm.best_period
    time_shift = period/2

    # initialise new passband sample data frame
    new_sample_cols = ['object_id','mjd','flux','flux_err','passband','detected']
    new_sample = pd.DataFrame(columns=new_sample_cols)

    for passb in range(0,6):
        # consider each passband separately
        pass_subset = subset[subset.passband==passb]

        # new dates to take samples for given passband
        new_mjd = list(pass_subset.mjd + time_shift)

        # assume the same probability of detection as in the original passband
        prob_detected = float(len(pass_subset[pass_subset.detected == 1])) / len(pass_subset)
        probs = [1-prob_detected, prob_detected]
        
        # generate new flux values by adding noise with variance equal to
        # the difference of true flux to predicted flux in the original passband
        pred_flux = lsm.predict(pass_subset.mjd, passb)
        true_flux = pass_subset.flux
        diff_flux = pred_flux - true_flux
        passb_var = np.mean(np.abs(diff_flux))
        noise = np.random.normal(0,passb_var,len(new_mjd))
        
        # use RandomForestRegressor to train a model for flux error
        flux_err_model = RandomForestRegressor(n_estimators=300)
        flux_err_cols = ['passband','flux', 'detected']
        x = pass_subset[flux_err_cols]
        x['noise'] = diff_flux
        x['phase'] = [(v % period) for v in pass_subset.mjd]
        flux_err_model.fit(x, pass_subset.flux_err)
        
        # create a data frame to store the new passband sample
        new_passb = pd.DataFrame(columns=new_sample_cols)
        new_passb['flux'] = lsm.predict(new_mjd, passb) + noise
        new_passb['mjd'] = new_mjd
        new_passb['passband'] = passb
        new_passb['object_id'] = new_id
        new_passb['detected'] = [np.random.choice(np.arange(0,2), p=probs) for date in new_mjd]
        
        # predict flux error and append to the data frame
        x_flux_err = new_passb[flux_err_cols]
        x_flux_err['noise'] = noise
        x_flux_err['phase'] = [(v % period) for v in new_passb.mjd]
        new_passb['flux_err'] = flux_err_model.predict(x_flux_err) 
        
        # add the new passband sample to parent object sample
        new_sample = new_sample.append(new_passb)

    # add the new sample to the augumented data frames
    new_series = new_series.append(new_sample, sort=True)
    new_sample_metadata = metadata[metadata.object_id==obj_id]
    new_sample_metadata['object_id'] = new_id
    new_metadata = new_metadata.append(new_sample_metadata)

    if int(count % (max_count/15)) == 0:
        enablePrint()
        progress = int(count*100.0/max_count)
        print('> galactic-progress: %i perc.' % progress)
        blockPrint()

enablePrint()
print('[1/2] Galactic data augumented.\n')

# augument extragalactic data by simply adding noise
extragal_objects = metadata[metadata.hostgal_photoz!=0].object_id
count, max_count = 0, len(extragal_objects)

for obj_id in extragal_objects:

    count += 1
    new_id = new_id + 1
    subset = series[series.object_id==obj_id]

    flux = list(subset.flux)
    flux_err = list(subset.flux_err)
    
    # add noise to the flux with variance equal to the flux error for that observation
    subset['flux'] = [np.random.normal(flux[i],flux_err[i],1)[0] for i in range(len(flux))]
    subset['object_id'] = new_id

    # add the new sample to the augumented data frames
    new_series = new_series.append(subset, sort=True)
    new_sample_metadata = metadata[metadata.object_id==obj_id]
    new_sample_metadata['object_id'] = new_id
    new_metadata = new_metadata.append(new_sample_metadata)

    if int(count % (max_count/10)) == 0:
        progress = int(count*100.0/max_count)
        print('> extragal-progress: %i perc.' % progress)

print('[2/2] Extragalactic data augumented.\n')

# save new data to csv files
print('Exporting files...')
new_metadata.to_csv('aug_set_metadata.csv', header=True, index=False)
new_series.to_csv('aug_set.csv', header=True, index=False)
print('Files exported. Execution finished.')
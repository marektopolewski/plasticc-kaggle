import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/modules/cs342/Assignment2/training_set.csv', usecols=['object_id','mjd','flux', 'passband'])
metadata = pd.read_csv('/modules/cs342/Assignment2/training_set_metadata.csv', usecols=['object_id','target'])

data = data.merge(metadata)
data = data[data.object_id!=74256178]         # bad sample

from scipy import optimize
def sine_func(x, a, b):
        return a * np.sin(b * x)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def data_aug(flux, mjd, gap_i):
    """
    Interpolate the flux to fill in the gaps in data.
    Add some noise to generat new series
    """
    sample_rate = (float(len(mjd[:gap_i[0]]))/(mjd[gap_i[0]]-mjd[0]))/3
    sample_size_gap = [
        int(sample_rate * (mjd[gap_i[0]+1]-mjd[gap_i[0]])),
        int(sample_rate * (mjd[gap_i[1]+1]-mjd[gap_i[1]]))
    ]

    mjd_pol = PolynomialFeatures(13).fit_transform([[x] for x in mjd])
    model = LinearRegression()
    model.fit(mjd_pol, flux)

#     params, params_covariance = optimize.curve_fit(sine_func, mjd, flux, p0=[2, np.max(flux)])
#     plt.scatter(mjd,flux)
#     whole_plot = np.linspace(np.min(mjd),np.max(mjd),100)
#     plt.plot(whole_plot, sine_func(whole_plot, params[0], params[1]))

#     whole_plot = np.linspace(np.min(mjd),np.max(mjd),20)
#     whole_plot_pol = PolynomialFeatures(13).fit_transform([[x] for x in whole_plot])
#     plt.scatter(mjd,flux)
#     plt.scatter(whole_plot, model.predict(whole_plot_pol))
#     plt.show()
#     plt.pause(2)
#     plt.clf()

    for i in range(0,len(gap_i)):
        gap_mjd = np.linspace(mjd[gap_i[i]], mjd[gap_i[i]+1], sample_size_gap[i])

        gap_mjd_pol = PolynomialFeatures(13).fit_transform([[x] for x in gap_mjd])
        gap_flux = model.predict(gap_mjd_pol)

#         gap_flux = sine_func(gap_mjd, params[0], params[1])

        noise = np.random.normal(0,np.std(flux)/5,sample_size_gap[i])
        gap_flux = list(np.array(gap_flux) + noise)

        flux = np.append(flux, gap_flux)
        mjd = np.append(mjd, gap_mjd)

    return (mjd,flux)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def data_aug(flux, mjd, gap_i):
    """
    Interpolate the flux to fill in the gaps in data.
    Add some noise to generat new series
    """
    sample_rate = (float(len(mjd[:gap_i[0]]))/(mjd[gap_i[0]]-mjd[0]))/3
    sample_size_gap = [
        int(sample_rate * (mjd[gap_i[0]+1]-mjd[gap_i[0]])),
        int(sample_rate * (mjd[gap_i[1]+1]-mjd[gap_i[1]]))
    ]

    mjd_pol = PolynomialFeatures(13).fit_transform([[x] for x in mjd])
    model = LinearRegression()
    model.fit(mjd_pol, flux)

#     params, params_covariance = optimize.curve_fit(sine_func, mjd, flux, p0=[2, np.max(flux)])
#     plt.scatter(mjd,flux)
#     whole_plot = np.linspace(np.min(mjd),np.max(mjd),100)
#     plt.plot(whole_plot, sine_func(whole_plot, params[0], params[1]))

#     whole_plot = np.linspace(np.min(mjd),np.max(mjd),20)
#     whole_plot_pol = PolynomialFeatures(13).fit_transform([[x] for x in whole_plot])
#     plt.scatter(mjd,flux)
#     plt.scatter(whole_plot, model.predict(whole_plot_pol))
#     plt.show()
#     plt.pause(2)
#     plt.clf()

    for i in range(0,len(gap_i)):
        gap_mjd = np.linspace(mjd[gap_i[i]], mjd[gap_i[i]+1], sample_size_gap[i])

        gap_mjd_pol = PolynomialFeatures(13).fit_transform([[x] for x in gap_mjd])
        gap_flux = model.predict(gap_mjd_pol)

#         gap_flux = sine_func(gap_mjd, params[0], params[1])

        noise = np.random.normal(0,np.std(flux)/5,sample_size_gap[i])
        gap_flux = list(np.array(gap_flux) + noise)

        flux = np.append(flux, gap_flux)
        mjd = np.append(mjd, gap_mjd)

    return (mjd,flux)


import heapq
count = 0
aug_data = pd.DataFrame(columns=['flux','mjd','passband','object_id','target'])
new_id = data.object_id.max()
for obj_id in data.object_id.unique():
    count += 1
    obj_data = data[data.object_id==obj_id]
    for passband in range(0,6):
        passb = obj_data[obj_data.passband == passband]

        dates = list(passb.mjd)
        gaps = [x - dates[i - 1] for i, x in enumerate(dates) if i > 0]
        gap_time = min(heapq.nlargest(2,gaps))

        big_gaps = list(np.where(np.array(gaps)>=gap_time)[0])
        if len(big_gaps) == 2 :
            res = data_aug(list(passb.flux), list(passb.mjd), big_gaps)
#             aug_data = aug_data.append(pd.DataFrame())

            plt.scatter(res[0], res[1])
            plt.scatter(list(passb.mjd),list(passb.flux))
            plt.show()
            plt.pause(5)
            plt.clf()

    if count % 500 == 0:
        print('covered %i' % count)

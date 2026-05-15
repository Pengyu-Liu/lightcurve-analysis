#period analysis
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from astropy.nddata import block_reduce
from lightcurve_reduction.special_functions import robust_sigma
from bgls import bgls
from tqdm import tqdm

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 13,
          #'legend.frameon': True,
          'figure.figsize': (9, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15}
pylab.rcParams.update(params)


#read in light curves saved from light_curve.py
target_name='J0200-5105'
date='2021_10_20'
in_dir='/Users/s2224823/data/SOFI/'+date+'/'+target_name+'/reduced'
# in_dir='/Users/s2224823/data/old_SOFI_data/myPSO318/reduced'
#choose aperture
#for 2021_10
apr_rad=np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
#for 2022_02
# apr_rad = np.array([3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.0])
apr_r = np.where(apr_rad == 4.5)[0][0]
#all curve: axis=0, aperture; axis=1, target (index=0)+reference stars
all_curves=np.load(in_dir+'/final_curves.npy', allow_pickle=True)
curves=all_curves[apr_r]
xjdh=np.loadtxt(in_dir+'/xjdh.txt')
refstar_ind=np.load(in_dir+'/ind_final_refstar.npy', allow_pickle=True)
refstar_ind=refstar_ind[apr_r]

#bin data
n_bin=1
xjdh=np.mean(xjdh.reshape(-1, n_bin), axis=1)
curves=block_reduce(curves, block_size=(1, n_bin), func=np.mean)
# print((xjdh[-1]-xjdh[0])/xjdh.size)
# exit()

#frequency grid of lomb scargle algorithm
freq_grid=np.logspace(np.log10(1/(2*xjdh.max())), np.log10(1/0.3), 1000)
periods=1/freq_grid

#the number of reference stars+target: 0 is the detrended curve of target
num=curves.shape[0]
rms=np.zeros(num)
power=np.zeros((num, freq_grid.size))
LS=[]
for i in range(num):
    #compute rms
    rms[i]=robust_sigma(np.roll(curves[i],-1)-curves[i])/np.sqrt(2)
    LS.append(LombScargle(xjdh, curves[i], rms[i], normalization='standard', nterms=1))
    #compute LS
    power[i] = LS[i].power(frequency=freq_grid, method='chi2')

#compute false alarm level by applying bootstrap to the target light curve
FAP1=LS[0].false_alarm_level(0.01, method='bootstrap')
FAP5=LS[0].false_alarm_level(0.05, method='bootstrap')
print('1% FAP calculated from the target curve: ', FAP1)

np.savetxt(in_dir+'/FAP_target_51.txt', np.array([FAP5, FAP1]))
#also save out detrended lightcurve and its error
detrend_curve = np.ones((curves[0].size, 2))
detrend_curve[:,0] *= curves[0]
detrend_curve[:,1] *= rms[0]

np.savetxt(in_dir + '/detrended_curve.txt', detrend_curve)

#compute FAP from random permuted lightcurve
# print('compute FAP by random permuting target lightcurve 1000 times with the assumption of null detection')
# peakPower_target=np.zeros(1000)
# for i in tqdm(range(peakPower_target.size)):
#         #randomCurve=np.random.permutation(curves[0])
#         #sample with replacements
#         randomCurve=np.random.choice(curves[0], size=curves[0].size)
#         power_ref = LombScargle(xjdh, randomCurve, rms[0], normalization='psd', nterms=1).power(
#             frequency = freq_grid, method='chi2')
#         #store the peak power
#         peakPower_target[i] = power_ref.max()
#
# FAP1_target=np.percentile(peakPower_target, 99)
# FAP5_target=np.percentile(peakPower_target, 95)
# print('1% FAP calculated from the random permuted target curve: ', FAP1_target)

# # plt.figure()
# # plt.plot(power[1]-power[2])
# # plt.savefig(in_dir+'/plots/test_curve.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# # exit()
#
# # compute FAP from reference light curves: simulate 1000 light curves for each reference star
# np.random.seed(11)
# peakPower=np.zeros((num-1, 1000))
# #correspond period of peakPower
# peakPowerPrd=np.zeros((num-1, 1000))
# count=0
# print('compute 1000 simulated light curves and peakPower for each reference star')
# # powerAll=np.zeros((1000*(num-1), periods.size))
# plt.figure()
# for i in tqdm(range(peakPower.shape[0])):
#     for j in range(peakPower.shape[1]):
#         #curves[i+1]: skip the target
#         randomCurve=np.random.permutation(curves[i+1])
#         #Because we random permute the original curve, we should not use rms in LS compute?
#         power_ref = LombScargle(xjdh, randomCurve, rms[i+1], normalization='psd', nterms=1).power(
#             frequency = freq_grid, method='chi2')
#         # powerAll[count]= np.copy(power_ref)
#         count+=1
#         #store the peak power
#         peakPower[i,j] = power_ref.max()
#         peakPowerPrd[i,j] = periods[np.argmax(power_ref)]
#         plt.plot(periods, power_ref, color='grey')

#sort flatten peakPower
# peakPowerSorted = np.sort( peakPower, axis=None )
# #95 % threshold
# level95=np.percentile(peakPower, 95)
# #99% threshold
# level99=np.percentile(peakPower, 99)
#95% and 99% envolope
# envolope95=np.percentile(powerAll, 95, axis=0)
# envolope99=np.percentile(powerAll, 99, axis=0)
# del powerAll
# print('95% threshold power level: ', level95)
# print('99% threshold power level: ', level99)

# np.savetxt(in_dir+'/FAP_target_ref_51.txt', np.array([level95, level99]))
#
# FAP=np.loadtxt(in_dir+'/FAP_target_ref_51.txt')
# level95=FAP[0]
# level99=FAP[1]
# #plot all random curves
# plt.xlabel('Period [hr]')
# plt.ylabel('Power')
# plt.xlim(0, periods.max())
# plt.hlines(level95,xmin=0,xmax=periods.max(),colors='green',linestyles='dashed', label='FAP = 5%')
# plt.hlines(level99,xmin=0,xmax=periods.max(),colors='brown',linestyles='dashed', label='FAP = 1%')
# # plt.plot(periods, envolope95, color='green', label='5% FAP envolope' )
# # plt.plot(periods, envolope99, color='brown', label='1% FAP envolope' )
# plt.legend()
# plt.minorticks_on()
# plt.tight_layout()
# plt.savefig(in_dir+'/plots/LS_simulatedwithError_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()

#window function
wind_t = np.linspace(0, np.max(xjdh), 500)
wind_x = np.concatenate((wind_t, curves[0]), axis=0)
wind_y = np.concatenate((np.zeros_like(wind_t), np.ones_like(curves[0])), axis=0)
wind_x1 = np.sort(wind_x)
wind_y1 = wind_y[np.argsort(wind_x)]
# freq_grid1=np.logspace(np.log10(1/10), np.log10(1/0.3), 2000)
# periods1=1/freq_grid1
window_power = LombScargle(xjdh, np.ones_like(curves[0]), normalization='standard', center_data=False, fit_mean=False, nterms=1).power(frequency=freq_grid)




periond_maxprob=periods[np.argmax(power[0])]
print('period with maximum power: {:.5f} hr'.format(periond_maxprob))

#plot target and reference stars periodogram
plt.figure()
plt.plot(periods, power[0], label='target')
for i in range(num-1):
    plt.plot(periods, power[i+1], label='{:}'.format(refstar_ind[i]))
plt.plot(periods, window_power, ls='--', c='black', label='LS window')
plt.xlabel('Period [hr]')
plt.ylabel('Power')
plt.legend()
#plt.ylabel('probability')
plt.xlim(0, periods.max())
# plt.hlines(level95,xmin=0,xmax=periods.max(),colors='green',linestyles='dashed', label='FAP = 5%')
# plt.hlines(level99,xmin=0,xmax=periods.max(),colors='brown',linestyles='dashed', label='FAP = 1%')
# plt.hlines(FAP5,xmin=0,xmax=periods.max(),colors='green',linestyles='dotted', label='self FAP=5%')
# plt.hlines(FAP1,xmin=0,xmax=periods.max(),colors='brown',linestyles='dotted', label='self FAP=1%')
plt.hlines(FAP5,xmin=0,xmax=periods.max(),colors='green',linestyles='dashed', label='FAP = 5%')
plt.hlines(FAP1,xmin=0,xmax=periods.max(),colors='brown',linestyles='dashed', label='FAP = 1%')
# plt.plot(periods, envolope95, color='green', label='5% FAP envolope' )
# plt.plot(periods, envolope99, color='brown', label='1% FAP envolope' )
plt.legend()
plt.minorticks_on()
plt.tight_layout()
plt.savefig(in_dir+'/plots/LS_curvewithError_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
plt.close()


#compute the sensitivity plot by injection
#first: low-order polynomial fitting
# fitModel=np.poly1d(np.polyfit(xjdh, curves[0], 2))
# fitCurve=fitModel(xjdh)
# #remove potential variable trend
# flatCurve=curves[0]/fitCurve
# #check fitting result
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
# ax1.plot(xjdh, curves[0], label='target curve')
# ax1.plot(xjdh, fitCurve, linestyle='dashed', label='poly fit')
# ax1.set_ylabel('Relative flux')
# ax1.legend()
# ax1.minorticks_on()
# ax2.plot(xjdh, flatCurve, label='flattened curve')
# ax2.set_xlabel('Elasped time [hr]')
# ax2.legend()
# ax2.minorticks_on()
# plt.savefig(in_dir+'/plots/fitting_bin{:}_apr{:}_Pmin05.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()
#
# np.random.seed(31)
# #random permute
# pmtCurve=np.random.permutation(flatCurve)
# #inject a sinusoidal signal
# amp=1e-2
# prd=3
# phase=2*np.pi*np.random.random_sample()
# print('injected amp, prd and phase: ', amp, prd, phase)
# injectCurve = pmtCurve + amp*np.sin(2 * np.pi * (1/prd) * xjdh + phase)
# #calculate new error in this way?
# error=robust_sigma(np.roll(injectCurve ,-1)-injectCurve)/np.sqrt(2)
# #compute LS power spectrum: use rms[0] or error? From rough test: no much difference
# retrvLS= LombScargle(xjdh, injectCurve,  rms[0], normalization='psd', nterms=1)
# retrvPower=retrvLS.power(frequency = freq_grid, method='chi2')
# best_freq=freq_grid[np.argmax(retrvPower)]
# LS_fit=retrvLS.model(xjdh, best_freq)
# #model parameter: a+bsin(2pift)+ccos(2pift)=a+rsin(2pift+phi), b=rcosphi. real model offset: a+retrvLS.offset()
# LS_best_model=retrvLS.model_parameters(best_freq)
# model_amp=np.sqrt(LS_best_model[1]**2+LS_best_model[2]**2)
# model_phase=np.arcsin(LS_best_model[2]/model_amp)
# print('Retrieved period: ', 1/best_freq)
# print('Retrieved amplitude: ', model_amp)
# print('Retrieved phase', model_phase)
# fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
# ax1.plot(xjdh, injectCurve, 'o')
# ax1.plot(xjdh, LS_fit, color='black', label='LS model')
# # double check calculated sine model
# # ax1.plot(xjdh, model_amp*np.sin(2*np.pi*best_freq*xjdh+model_phase)+LS_best_model[0]+retrvLS.offset())
# ax1.set_xlabel('Elasped time [hr]')
# ax1.set_ylabel('Relative flux')
# ax1.legend()
# ax1.minorticks_on()
# ax2.plot(periods, retrvPower)
# ax2.vlines(1/best_freq, ymin=0, ymax=retrvPower.max(), linestyles='dashed')
# ax2.set_xlabel('Period [hr]')
# ax2.set_ylabel('Power')
# ax2.minorticks_on()
# plt.savefig(in_dir+'/plots/injectionCheck_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()


#compute the window transform
# window=np.ones_like(target_curve)
# window_power = LombScargle(xjdh, window, normalization='psd', center_data='False', fit_mean='False', nterms=1).power(
#                     frequency=freq_grid, method='chi2')
# plt.figure()
# plt.plot(periods, window_power)
# plt.xlabel('period [hr]')
# plt.ylabel('power')
# #plt.ylabel('probability')
# #plt.xlim(0,0.2)
# plt.minorticks_on()
# plt.savefig(in_dir+'/plots/LS_window_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()

#compute LS of reference light curves
#the number of reference stars
# refN=curves.shape[0]-1
# refRMS=np.zeros(refN)
# refpower=np.zeros((refN, freq_grid.size))
# refLS=[]
# for i in range(refN):
#     refRMS[i]=robust_sigma(np.roll(curves[i+1],-1)-curves[i+1])/np.sqrt(2)
#     refLS.append(LombScargle(xjdh, curves[i+1], refRMS[0], normalization='psd', nterms=1))
#     refpower[i] = refLS[i].power(frequency=freq_grid, method='chi2')
#
# plt.figure()
# for i in range(refN):
#     plt.plot(periods, refpower[i],'o')
# plt.xlabel('period [hr]')
# plt.ylabel('power')
# plt.savefig(in_dir+'/plots/LS_ref_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)

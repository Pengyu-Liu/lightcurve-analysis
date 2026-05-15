#compute sensitivity plot by injection
import multiprocessing
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.timeseries import LombScargle
from astropy.nddata import block_reduce
from lightcurve_reduction.special_functions import robust_sigma
from scipy.optimize import curve_fit
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

def sincurve(x, a, b, c):
    return 1+a*np.sin(2*np.pi*b*x+c)

def inject_one_position(xy_position, amp, prd, xjdh, freq_grid, flatCurve, rms, Ncurve, amp_thres, prd_thres, fap):
    count=0
    amp_one_position=amp[xy_position[0]]
    prd_one_position=prd[xy_position[1]]
    for k in range(Ncurve):
        #random permute
        pmtCurve=np.random.permutation(flatCurve)
        phase=2*np.pi*np.random.random_sample()
        injectCurve = pmtCurve + amp_one_position*np.sin(2 * np.pi * (1/prd_one_position) * xjdh + phase)
        #compute LS power spectrum
        retrvLS= LombScargle(xjdh, injectCurve, rms, normalization='psd', nterms=1)
        retrvPower=retrvLS.power(frequency = freq_grid, method='chi2')
        best_freq=freq_grid[np.argmax(retrvPower)]
        maxPower=np.max(retrvPower)
        LS_fit=retrvLS.model(xjdh, best_freq)
        #model parameter: a+bsin(2pift)+ccos(2ppift)=a+rsin(2pift+phi), b=rcosphi. real model offset: a+retrvLS.offset()
        LS_best_model=retrvLS.model_parameters(best_freq)
        model_amp=np.sqrt(LS_best_model[1]**2+LS_best_model[2]**2)
        model_phase=np.arcsin(LS_best_model[2]/model_amp)
        model_prd=1/best_freq
        #amplitude criterion
        crit1=abs(model_amp-amp_one_position)/amp_one_position < amp_thres
        #period criterion
        crit2=abs(model_prd-prd_one_position)/prd_one_position< prd_thres
        #loose period criterion for long period
        # if prd_one_position > 2*xjdh.max():
        #     if model_prd > 2*xjdh.max():
        #         crit2=True
        #     else:
        #         crit2=False
        # else:
        #     crit2=abs(model_prd-prd_one_position)/prd_one_position< prd_thres

        #fap criterion
        crit3=maxPower>fap
        if crit1*crit2*crit3:
            count+=1
    return count/Ncurve

def injection(use_multiprocess, ncpus, amp, prd, xjdh, freq_grid, flatCurve, rms, Ncurve, amp_thres, prd_thres, fap):
    sens_map=np.ones((amp.size, prd.size))
    #coordinates array
    coords=np.argwhere(sens_map)
    if use_multiprocess:
        multiprocess_inject_position = partial(
            inject_one_position, amp=amp, prd=prd, xjdh=xjdh,
            freq_grid=freq_grid, flatCurve=flatCurve, rms=rms,
            Ncurve=Ncurve, amp_thres=amp_thres, prd_thres=prd_thres, fap=fap
        )
        if __name__ == '__main__':
            pool = Pool(processes=ncpus)
            for idx, result in enumerate(tqdm(pool.imap(multiprocess_inject_position, coords), total=len(coords))):
                sens_map[coords[idx][0], coords[idx][1]] = result
            del result
            del pool
    else:
        #no multiprocessing
        for idx in tqdm(range(len(coords))):
            sens_map[coords[idx][0], coords[idx][1]] = inject_one_position(coords[idx],
                amp=amp, prd=prd, xjdh=xjdh,
                freq_grid=freq_grid, flatCurve=flatCurve, rms=rms,
                Ncurve=Ncurve, amp_thres=amp_thres, prd_thres=prd_thres, fap=fap
            )

    return sens_map

#read in light curves saved from light_curve.py
target_name='J2323-0152'
date='Oct24'
in_dir='/Users/data/SOFI/'+date+'/'+target_name+'/reduced'
# in_dir='/Users/s2224823/data/old_SOFI_data/myPSO318/reduced'
#choose aperture
apr_rad=np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
apr_r=0
#all curve: axis=0, aperture; axis=1, target (index=0)+reference stars
all_curves=np.load(in_dir+'/final_curves.npy', allow_pickle=True)
curves=all_curves[apr_r]
xjdh=np.loadtxt(in_dir+'/xjdh.txt')
refstar_ind=np.load(in_dir+'/ind_final_refstar.npy', allow_pickle=True)
refstar_ind=refstar_ind[apr_r]

#read in FAP computed from simulated reference lightcurves: [level95, level99]]
FAP=np.loadtxt(in_dir+'/FAP_target_51.txt')
FAP1=FAP[1]

#bin data
n_bin=1
xjdh=np.mean(xjdh.reshape(-1, n_bin), axis=1)
curves=block_reduce(curves, block_size=(1, n_bin), func=np.mean)
# print((xjdh[-1]-xjdh[0])/xjdh.size)
# exit()

#frequency grid of lomb scargle algorithm
freq_grid=np.logspace(np.log10(1/30), np.log10(1/0.3), 1000)
periods=1/freq_grid
rms=robust_sigma(np.roll(curves[0],-1)-curves[0])/np.sqrt(2)

#compute the sensitivity plot by injection
#first: low-order polynomial fitting
fitModel=np.poly1d(np.polyfit(xjdh, curves[0], 2))
fitCurve=fitModel(xjdh)
#remove potential variable trend
flatCurve=curves[0]/fitCurve

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
# plt.savefig(in_dir+'/plots/fitting_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()
#
# np.random.seed(31)
# #random permute
# pmtCurve=np.random.permutation(flatCurve)
# #inject a sinusoidal signal
# amp=5e-2
# prd=20
# phase=2*np.pi*np.random.random_sample()
# print('injected amp, prd and phase: ', amp, prd, phase)
# injectCurve = pmtCurve + amp*np.sin(2 * np.pi * (1/prd) * xjdh + phase)
# #calculate new error in this way?
# error=robust_sigma(np.roll(injectCurve ,-1)-injectCurve)/np.sqrt(2)
# #compute LS power spectrum: use rms[0] or error? From rough test: no big difference
# retrvLS= LombScargle(xjdh, injectCurve,  rms, normalization='psd', nterms=1)
# retrvPower=retrvLS.power(frequency = freq_grid, method='chi2')
# best_freq=freq_grid[np.argmax(retrvPower)]
# maxPower=np.max(retrvPower)
# LS_fit=retrvLS.model(xjdh, best_freq)
# #model parameter: a+bsin(2pift)+ccos(2pift)=a+rsin(2pift+phi), b=rcosphi. real model offset: a+retrvLS.offset()
# LS_best_model=retrvLS.model_parameters(best_freq)
# model_amp=np.sqrt(LS_best_model[1]**2+LS_best_model[2]**2)
# model_phase=np.arcsin(LS_best_model[2]/model_amp)
# #or fit sine curve using scipy
# #curve fitting from scipy
# # pop, pcov = curve_fit(sincurve, xjdh, injectCurve, p0=[1e-2, best_freq, 0], sigma=rms*np.ones_like(xjdh), absolute_sigma=True)
# # print(pop)
# # print(1/pop[1])
#
# print('Retrieved period: ', 1/best_freq)
# print('Retrieved amplitude: ', model_amp)
# print('Retrieved phase', model_phase)
# fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
# ax1.plot(xjdh, injectCurve, 'o')
# ax1.plot(xjdh, injectCurve-pmtCurve+1, label='injected signal P={:}'.format(prd))
# ax1.plot(xjdh, LS_fit, color='black', label='LS model')
# # ax1.plot(xjdh, sincurve(xjdh, *pop), label='scipy fitting')
# # double check calculated sine model
# # ax1.plot(xjdh, model_amp*np.sin(2*np.pi*best_freq*xjdh+model_phase)+LS_best_model[0]+retrvLS.offset())
# ax1.set_xlabel('Elasped time [hr]')
# ax1.set_ylabel('Relative flux')
# ax1.legend()
# ax1.minorticks_on()
# ax2.plot(periods, retrvPower)
# ax2.vlines(1/best_freq, ymin=0, ymax=retrvPower.max(), linestyles='dashed')
# ax2.hlines(FAP1,xmin=0,xmax=periods.max(),colors='brown',linestyles='dashed', label='1% FAP')
# ax2.set_xlabel('Period [hr]')
# ax2.set_ylabel('Power')
# ax2.legend()
# ax2.minorticks_on()
# plt.savefig(in_dir+'/plots/injectionCheck_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
# plt.close()


#compute senstivity plot
#inject fake signals and retrieve them
np.random.seed(31)
#gid number in amp and period
grd=100
#number of simulated curves at each grid
Ncurve=100
#amplitude and period grids
amp_grid=np.linspace(0.005, 0.1, grd)
prd_grid=np.linspace(1.5, 20, grd)
#threshold to decide if injected siganl is retrieved
amp_thres=0.3
prd_thres=0.3
#whether use mutiprocessing
use_multiprocess=True
#number of cores
ncpus=7
#compute sensitivity map
sens_map=injection(use_multiprocess=use_multiprocess, ncpus=ncpus,
                  amp=amp_grid, prd=prd_grid, xjdh=xjdh, freq_grid=freq_grid,
                  flatCurve=flatCurve, rms=rms,
                  Ncurve=Ncurve, amp_thres=amp_thres, prd_thres=prd_thres, fap=FAP1)

np.savetxt(in_dir+'/sensitivity_map_prdthres_03.txt', sens_map)
np.savetxt(in_dir+'/prd_grid.txt', prd_grid)
np.savetxt(in_dir+'/amp_grid.txt', amp_grid)
np.savetxt(in_dir+'/freq_grid.txt', freq_grid)

# #make a plot of senstivity map
# X, Y = np.meshgrid(prd_grid, amp_grid)
# levels = np.array([0.25, 0.50, 0.75, 0.9, 0.95])
# # norm = cm.colors.Normalize(0, 1)
# cmap = cm.RdBu
# extent=(prd_grid.min(), prd_grid.max(), amp_grid.min()*100, amp_grid.max()*100)
# fig, ax=plt.subplots(constrained_layout=True)
# im=ax.imshow(sens_map, vmax=1, vmin=0, cmap=cm.RdBu, aspect='auto', extent=extent)
# # cs=ax.contourf(X, Y, sensit, levels, norm=norm, cmap=cm.get_cmap(cmap))
# #plot contour lines
# cs=ax.contour(sens_map, levels, colors='black', linewidths=1.5, extent=extent)
# #label contour lines
# ax.clabel(cs, inline=True, fontsize=10)
# ax.set_xlabel('Period [hr]')
# ax.set_ylabel('Amplitude [%]')
# cbar=fig.colorbar(im, ax=ax)
# plt.savefig(in_dir+'/plots/sensitivity.png', dpi=200)
# plt.close()
#

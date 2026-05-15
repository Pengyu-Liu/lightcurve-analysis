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
target_name='W1636-0743Js'
date='2023_05_10'
in_dir='/Users/s2224823/data/SOFI/'+date+'/'+target_name+'/reduced'
#choose aperture
#for 2021_10:
# apr_rad=np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
#for 2022_02
apr_rad = np.array([3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.0, 7.5, 8.0, 8.5, 9])
apr_r = np.where(apr_rad ==3.5)[0][0]
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


#Try Bayesian Generalized Lomb Scargle algorithm
rms_target=robust_sigma(np.roll(curves[0],-1)-curves[0])/np.sqrt(2)
rms_target=np.ones_like(curves[0])*rms_target
periods, power = bgls(xjdh, curves[0], rms_target, plow=0.2, phigh= 2*xjdh.max() , ofac=10)

periond_maxprob=periods[np.argmax(power)]
print('period with maximum power: {:.5f} hr'.format(periond_maxprob))

#plot target and reference stars periodogram
plt.figure()
plt.plot(periods, power)
plt.xlim(0, periods.max())
plt.vlines(periond_maxprob, ymin=0, ymax=np.max(power), colors='black',linestyles='dashed', label='Max likelihood P={:.2f}hr'.format(periond_maxprob))
plt.xlabel('Period [hr]')
plt.ylabel('Probability')
plt.legend()
plt.minorticks_on()
plt.tight_layout()
plt.savefig(in_dir+'/plots/BGLS_curvewithError_bin{:}_apr{:}.png'.format(n_bin, apr_rad[apr_r]), dpi=200)
plt.close()

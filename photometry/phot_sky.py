#measure sky variation
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import QTable
import glob
from photutils.aperture import CircularAperture,aperture_photometry
from tqdm import tqdm
from lightcurve_reduction.special_functions import aper_sub_sky

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (8, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

target_name='J0044+0228'
date='Oct20'
in_dir='/Users/data/SOFIP108/'+date+'/'+target_name+'/xtalk'
out_dir='/Users/data/SOFIP108/'+date+'/'+target_name+'/reduced'
path=sorted(glob.glob(in_dir+'/*.fits'))
n_files=len(path)

x=514
y=354
positions = [(x, y)]
apr_rad=np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
apertures = [CircularAperture(positions, r=r) for r in apr_rad]
phot_all=np.zeros((n_files, apr_rad.size+1))
name=['JD']
for i in tqdm(range(n_files)):
    file, header=fits.getdata(path[i],header=True)
    phot_all[i,0]=header['MJD-OBS']
    phot_table = aperture_photometry(file, apertures)
    for j in range(apr_rad.size):
        phot_all[i,j+1]=phot_table['aperture_sum_{:}'.format(j)]
for j in range(apr_rad.size):
    name.append('aperture_sum_{:}'.format(j))

t = QTable(phot_all,
           names=name,
          meta={'name': target_name})
ascii.write(t, out_dir+'/photometry_sky.dat', overwrite=True)

#plot out the sky variations
norm_sky = phot_all[:,1:]/np.median(phot_all[:,1:], axis=0)
plt.figure()
plt.plot((phot_all[:,0]-phot_all[0,0])*24, norm_sky[:,1],'-o')
plt.minorticks_on()
plt.xlabel('Elapsed time[h]')
plt.ylabel('relative sky flux')
plt.savefig(out_dir+'/plots/sky_flux_xtalk.png', dpi=200)

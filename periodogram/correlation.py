#explore the correlation of sky condition with lightcurves
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import seaborn as sn

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 15,
          'figure.figsize': (9, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)


target_name='2M0619-2903JsHKs'
date='2022_02_19'
in_dir='/Users/data/SOFI/'+date+'/'+target_name+'/reduced_H'
#load photometry data
#axis=0, files; axis=1, stars
xy_found=np.load(in_dir+'/xy_found.npy', allow_pickle=True)
#for 2021-10
#apr_rad=np.array([3.5, 4, 4.5, 5, 5.5, 6, 6.5])
#for 2022-02
apr_rad = np.array([3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.0])
apr_r=2
#raw curve: #axis=0, files; axis=1, stars; axis=2, aperture rad; axis=3, pho, bkg, pho-bkg
flux_fixed=np.load(in_dir+'/flux_fixed.npy')
#axis=0, files; axis=1, stars
peak_flux=np.load(in_dir+'/peak_flux.npy')
#all curve: axis=0, aperture; axis=1, target (index=0)+reference stars
all_curves=np.load(in_dir+'/final_curves.npy', allow_pickle=True)
curves=all_curves[apr_r]
#read in selected reference stars
refstar_ind=np.load(in_dir+'/ind_final_refstar.npy', allow_pickle=True)
refstar_ind=refstar_ind[apr_r]
#'JD', 'airmass', 'dimm_fwhm','median_fwhm', 'nods'
files_info=ascii.read(in_dir+'/files_info.dat')
#delete bad frames if have
# good_mask=np.ones(xy_found.shape[0], dtype='bool')
# bad_ind=[i for i in range(12)]
# good_mask[bad_ind]=False
# xy_found=xy_found[good_mask]
# flux_fixed=flux_fixed[good_mask]
# peak_flux=peak_flux[good_mask]
# files_info.remove_rows(bad_ind)

x_jd = np.loadtxt(in_dir + '/xjdh.txt')

#arcsec per pixel
pixel_scale=0.288
seeing=pixel_scale*files_info['median_fwhm'].data
airmass=files_info['airmass'].data

# print(files_info['median_fwhm'].data)
# exit()
#below from light_curve.py
#all curve: axis=0, aperture; axis=1, target (index=0)+reference stars
all_curves=np.load(in_dir+'/final_curves.npy', allow_pickle=True)
detrend_curves=all_curves[apr_r]
#read in selected reference stars
refstar_ind=np.load(in_dir+'/ind_final_refstar.npy', allow_pickle=True)
refstar_ind=refstar_ind[apr_r]

#find target
xtarg=414
ytarg=384
ind_targ=np.where((xy_found[0,:,0]>(xtarg-10))*(xy_found[0,:,0]<xtarg+10)*(xy_found[0,:,1]>(ytarg-10))*(xy_found[0,:,1]<ytarg+10))[0]
if ind_targ.size ==0:
    print('stop: target not found!')
    exit()

elif ind_targ.size>1:
    print('stop: multiple objects at target location!')
    exit()

#no local sky subtraction: 0; with local sky subtraction: 2
flux_ind = 0
#axis=0, stars; axis=1, aperture
#normalize by median brightness of different nods: A nod by A nod, B nod by B nod
medd_flux_fixed=np.zeros((flux_fixed.shape[0],flux_fixed.shape[1],flux_fixed.shape[2]))
ind_Anod=files_info['nods'].data== 0
ind_Bnod=files_info['nods'].data== 1
for i in range(flux_fixed.shape[1]):
    medflux_A=np.median(flux_fixed[ind_Anod,i,:, flux_ind],axis=0)
    medflux_B=np.median(flux_fixed[ind_Bnod,i,:, flux_ind],axis=0)
    for j in range(flux_fixed.shape[2]):
        medd_flux_fixed[ind_Anod,i,j]=flux_fixed[ind_Anod,i,j, flux_ind]/medflux_A[j]
        medd_flux_fixed[ind_Bnod,i,j]=flux_fixed[ind_Bnod,i,j, flux_ind]/medflux_B[j]

        #correct baseline of different nods
        num = np.min([np.sum(ind_Anod),np.sum(ind_Bnod)])
        shift1=np.median(medd_flux_fixed[ind_Bnod, i, j][0:num]-medd_flux_fixed[ind_Anod,i,j][0:num])
        medd_flux_fixed[ind_Anod,i,j]+=shift1



#plot the correlation of normalized flux and fwhm
fig, axs = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
axs[0,0].plot(medd_flux_fixed[:, ind_targ[0], apr_r], seeing,'o', label='target')
for i in range(len(refstar_ind)):
     axs[0,0].plot(medd_flux_fixed[:, refstar_ind[i], apr_r], seeing, 'o', label='ref star {:}'.format(refstar_ind[i]))
axs[0,0].set_xlabel('Normalized flux')
axs[0,0].set_ylabel('Seeing [arcsec]')
axs[0,0].legend()
axs[0,0].minorticks_on()

#plot the correlation of detrened flux and seeing
corr, pval=spearmanr(curves[0], seeing, nan_policy='omit')
axs[0,1].plot(curves[0], seeing,'o', label=r'target $\rho$={:.2f}'.format(corr))
for i in range(len(refstar_ind)):
     corr, pval=spearmanr(curves[i+1], seeing, nan_policy='omit')
     axs[0,1].plot(curves[i+1], seeing, 'o', label=r'ref star {:} $\rho$={:.2f}'.format(refstar_ind[i], corr))

axs[0,1].set_xlabel('Residual flux')
# axs[1].set_ylabel('Seeing [arcsec]')
axs[0,1].legend(bbox_to_anchor=(0.85, 0.7))
axs[0,1].minorticks_on()

#plot airmass and flux
axs[1,0].plot(medd_flux_fixed[:, ind_targ[0], apr_r], airmass,'o', label='target')
for i in range(len(refstar_ind)):
     axs[1,0].plot(medd_flux_fixed[:, refstar_ind[i], apr_r], airmass, 'o', label='ref star {:}'.format(refstar_ind[i]))

axs[1,0].set_xlabel('Normalized flux')
axs[1,0].set_ylabel('Airmass')
# axs[1,0].legend()
axs[1,0].minorticks_on()

#plot the correlation of detrened flux and airmass
corr, pval=spearmanr(curves[0], airmass, nan_policy='omit')
axs[1,1].plot(curves[0], airmass,'o', label=r'target $\rho$={:.2f}'.format(corr))
for i in range(len(refstar_ind)):
     corr, pval=spearmanr(curves[i+1], airmass, nan_policy='omit')
     axs[1,1].plot(curves[i+1], airmass, 'o', label=r'ref star {:} $\rho$={:.2f}'.format(refstar_ind[i], corr))

axs[1,1].set_xlabel('Residual flux')
# axs[1].set_ylabel('Seeing [arcsec]')
axs[1,1].legend(bbox_to_anchor=(0.85, 0.7))
axs[1,1].minorticks_on()

plt.savefig(in_dir+'/plots/seeAir_norResFlux.png', dpi=200)
plt.close()


#plot the correlation of raw flux and fwhm
fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
axs[0,0].plot(flux_fixed[:, ind_targ[0], apr_r, 2], seeing,'o', label='target')
for i in range(len(refstar_ind)):
     axs[0,0].plot(flux_fixed[:, refstar_ind[i], apr_r, 2], seeing, 'o', label='ref star {:}'.format(refstar_ind[i]))

axs[0,0].set_xlabel('Raw flux')
axs[0,0].set_ylabel('Seeing [arcsec]')
axs[0,0].legend()
axs[0,0].minorticks_on()
#plot the correlation of peak flux and seeing
axs[0,1].plot(peak_flux[:, ind_targ[0]], seeing,'o', label='target')
for i in range(len(refstar_ind)):
     axs[0,1].plot(peak_flux[:, refstar_ind[i]], seeing, 'o', label='ref star {:}'.format(refstar_ind[i]))

axs[0,1].set_xlabel('Peak flux')
# axs[1].set_ylabel('Seeing [arcsec]')
# axs[1].legend()
axs[0,1].minorticks_on()

#plot the correlation of raw flux and airmass
axs[1,0].plot(flux_fixed[:, ind_targ[0], apr_r, 2], airmass,'o', label='target')
for i in range(len(refstar_ind)):
     axs[1,0].plot(flux_fixed[:, refstar_ind[i], apr_r, 2], airmass, 'o', label='ref star {:}'.format(refstar_ind[i]))

axs[1,0].set_xlabel('Raw flux')
axs[1,0].set_ylabel('Airmass')
# axs[1,0].legend()
axs[1,0].minorticks_on()
#plot the correlation of peak flux and airmass
axs[1,1].plot(peak_flux[:, ind_targ[0]], airmass,'o', label='target')
for i in range(len(refstar_ind)):
     axs[1,1].plot(peak_flux[:, refstar_ind[i]], airmass, 'o', label='ref star {:}'.format(refstar_ind[i]))

axs[1,1].set_xlabel('Peak flux')
# axs[1].set_ylabel('Seeing [arcsec]')
# axs[1].legend()
axs[1,1].minorticks_on()
plt.savefig(in_dir+'/plots/seeAir_rawPeakFlux.png', dpi=200)
plt.close()

#plot the correlation of x,y drift
y_drift=np.zeros((xy_found.shape[0],xy_found.shape[1]))
x_drift=np.zeros((xy_found.shape[0],xy_found.shape[1]))
total_drift=np.zeros((xy_found.shape[0],xy_found.shape[1]))
firstAind=np.argwhere(ind_Anod==True).reshape(-1)[0]
firstBind=np.argwhere(ind_Bnod==True).reshape(-1)[0]
#correct baseline in y
y_drift[ind_Anod,:]=xy_found[ind_Anod,:,1]-xy_found[firstAind,:,1]
y_drift[ind_Bnod,:]=xy_found[ind_Bnod,:,1]-xy_found[firstBind,:,1]
x_drift[ind_Anod,:]=xy_found[ind_Anod,:,0]-xy_found[firstAind,:,0]
x_drift[ind_Bnod,:]=xy_found[ind_Bnod,:,0]-xy_found[firstBind,:,0]
total_drift=np.sqrt(x_drift**2+y_drift**2)

#plot xy drift
plt.figure()
plt.scatter(x_drift[:, ind_targ[0]], y_drift[:, ind_targ[0]], marker='o',label='target')
for i in range(len(refstar_ind)):
     plt.scatter(x_drift[:, refstar_ind[i]], y_drift[:, refstar_ind[i]], marker='o', label='ref star {:}'.format(refstar_ind[i]))
plt.ylabel('Y drift [pixel]')
plt.xlabel('X drift [pixel]')
plt.legend()
plt.minorticks_on()
plt.tight_layout()
plt.savefig(in_dir+'/plots/drift_corr.png', dpi=200)
plt.close()

#plot the correlation of drift with normalized and residual flux
fig, axs = plt.subplots(1, 2, figsize=(12, 7), sharey=True, constrained_layout=True)
axs[0].plot(medd_flux_fixed[:, ind_targ[0], apr_r], total_drift[:, ind_targ[0]],'o', label='target')
for i in range(len(refstar_ind)):
     axs[0].plot(medd_flux_fixed[:, refstar_ind[i], apr_r], total_drift[:, refstar_ind[i]], 'o', label='ref star {:}'.format(refstar_ind[i]))
axs[0].set_xlabel('Normalized flux')
axs[0].set_ylabel('Drift [pixel]')
axs[0].legend()
axs[0].minorticks_on()

#plot the correlation of detrended flux and drift
corr, pval=spearmanr(curves[0], total_drift[:, ind_targ[0]], nan_policy='omit')
axs[1].plot(curves[0], total_drift[:, ind_targ[0]],'o', label=r'target $\rho$={:.2f}'.format(corr))
for i in range(len(refstar_ind)):
     corr, pval=spearmanr(curves[i+1], total_drift[:, refstar_ind[i]], nan_policy='omit')
     axs[1].plot(curves[i+1], total_drift[:, refstar_ind[i]], 'o', label=r'ref star {:} $\rho$={:.2f}'.format(refstar_ind[i], corr))

axs[1].set_xlabel('Residual flux')
# axs[1].set_ylabel('Seeing [arcsec]')
axs[1].legend(bbox_to_anchor=(0.85, 0.7))
axs[1].minorticks_on()
plt.savefig(in_dir+'/plots/drift_flux.png', dpi=200)
plt.close()

#plot the correlation of airmass and seeing
plt.figure()
plt.plot(total_drift[:, ind_targ[0]], seeing,'o', label='target')
plt.xlabel('Drift [pixel]')
plt.ylabel('Seeing [arcsec]')
plt.legend()
plt.tight_layout()
plt.minorticks_on()
plt.savefig(in_dir+'/plots/seeDrift.png', dpi=200)
plt.close()

#compute residual flux spearmanr correlation between starsnames=['target']
corr_matrix_star, pval=spearmanr(curves, axis=1, nan_policy='omit')
names=['target']
for i in range(len(refstar_ind)):
    names.append('ref {:}'.format(refstar_ind[i]))
plt.figure()
ax=sn.heatmap(corr_matrix_star, vmin=-1, vmax=1, annot=True, fmt='.2f', cbar_kws={'label':'Spearman coefficient'}, xticklabels=names, yticklabels=names)
plt.tight_layout()
plt.savefig(in_dir+'/plots/corre_resiFlux_stars.png', dpi=200)
plt.close()

#compute raw flux spearmanr correlation between stars
ind=[ind_targ[0]]
for i in refstar_ind:
    ind.append(i)
corr_matrix_star, pval=spearmanr(medd_flux_fixed[:, ind,apr_r], axis=0, nan_policy='omit')
plt.figure()
ax=sn.heatmap(corr_matrix_star, vmin=-1, vmax=1, annot=True, fmt='.2f', cbar_kws={'label':'Spearman coefficient'}, xticklabels=names, yticklabels=names)
plt.tight_layout()
plt.savefig(in_dir+'/plots/corre_normFlux_stars.png', dpi=200)
plt.close()

# plt.figure()
# plt.plot(curves[3], curves[2], 'o')
# plt.xlabel('Normalized flux of ref 7')
# plt.ylabel('Normalized flux of ref 3')
# plt.tight_layout()
# plt.savefig(in_dir+'/plots/corre_refstars.png', dpi=200)
# plt.close()
#
# print(pearsonr(curves[3], curves[2]))

# #plot the correlation of normalized flux and fwhm
# plt.figure()
# plt.plot(medd_flux_fixed[:, ind_targ[0], apr_r], seeing,'o', label='target')
# for i in range(len(refstar_ind)):
#      plt.plot(medd_flux_fixed[:, refstar_ind[i], apr_r], seeing, 'o', label='ref star {:}'.format(refstar_ind[i]))
#
# plt.xlabel('Normalized flux')
# plt.ylabel('Seeing [arcsec]')
# plt.legend()
# plt.minorticks_on()
# plt.tight_layout()
# plt.savefig(in_dir+'/plots/seeing_normalFlux.png', dpi=200)
# plt.close()

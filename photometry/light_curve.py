import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.stats import sigma_clipped_stats,  sigma_clip
from astropy.modeling import models, fitting
from astropy.nddata import block_reduce
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from lightcurve_reduction.special_functions import robust_sigma
mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (9, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)


def choose_ref_stars(std_stars, slope_stars, ind_refstar, std=2, slope=2):
    #exclude references with large noise
    #[0] contains the std and slope of the targets
    ind_temp1=(std_stars[1:]<std*std_stars[0])
    ind_temp2=(abs(slope_stars[1:])<slope*abs(slope_stars[0]))
    ind_temp=ind_temp1&ind_temp2
    #return the index of the chosen star in original xy_found array
    return ind_refstar[ind_temp]

def curve_calibration(norm_flux, ind_ref, ind_star):
    ind_temp= ind_ref[ind_ref!=ind_star]
    calib_curve=np.median(norm_flux[:,ind_temp],axis=1)
    targ_calib=norm_flux[:,ind_star]/calib_curve
    return targ_calib, calib_curve

def robust_linear_fit(x, y, niter=3, sigma=3):
    fit = fitting.LinearLSQFitter()
    # initialize the outlier removal fitter
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    # initialize a linear model
    line_init = models.Linear1D()
    # fit the data with the fitter
    fitted_line, mask = or_fit(line_init, x, y)
    return fitted_line.slope.value

#-----------------------------------------------------------------
target_name='J0819-0335'
date='2022_02_10'
in_dir='/Users/data/SOFI/'+date+'/'+target_name+'/reduced'
#load photometry data
#axis=0, files; axis=1, stars
xy_found=np.load(in_dir+'/xy_found.npy')
#axis=0, files; axis=1, stars; axis=2, aperture rad; axis=3, pho, bkg, pho-bkg
flux_fixed=np.load(in_dir+'/flux_fixed.npy')
flux_scaled=np.load(in_dir+'/flux_scaled.npy')
apr_rad_fixed=np.array([3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.0])
apr_rad_scaled=np.array([0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5])
#axis=0, files; axis=1, stars
peak_flux=np.load(in_dir+'/peak_flux.npy')
#'JD', 'airmass', 'dimm_fwhm','median_fwhm', 'nods'
files_info=ascii.read(in_dir+'/files_info.dat')
n_file=xy_found.shape[0]

image_dir=in_dir.replace('reduced','skysub')
path=sorted(glob.glob(image_dir+'/*skysub.fits'))
#delete bad frames if have
# good_mask=np.ones(xy_found.shape[0], dtype='bool')
# bad_ind=[147, 148, 149]
# good_mask[bad_ind]=False
# xy_found=xy_found[good_mask]
# flux_scaled=flux_scaled[good_mask]
# flux_fixed=flux_fixed[good_mask]
# peak_flux=peak_flux[good_mask]
# files_info.remove_rows(bad_ind)
# for index in sorted(bad_ind, reverse=True):
#     del path[index]

#read from the header file MJD_OBS from the first image for multiple band data
# path_jd=sorted(glob.glob('/Users/s2224823/data/SOFI/'+date+'/'+target_name+'/xtalk/*.fits'))
# _, hd_id = fits.getdata(path_jd[0], header = True)
# jd_start= hd_id['MJD-OBS']
# x_jd=files_info['JD'].data
# x_jdh=(x_jd-jd_start)*24
# np.savetxt(in_dir+'/xjdh.txt',x_jdh)

x_jd=files_info['JD'].data
x_jdh=(x_jd-x_jd[0])*24
np.savetxt(in_dir+'/xjdh.txt',x_jdh)


#find target
xtarg=418
ytarg=387
ind_targ=np.where((xy_found[0,:,0]>(xtarg-10))*(xy_found[0,:,0]<xtarg+10)*(xy_found[0,:,1]>(ytarg-10))*(xy_found[0,:,1]<ytarg+10))[0]
if ind_targ.size ==0:
    print('stop: target not found!')
    exit()

elif ind_targ.size>1:
    print('stop: multiple objects at target location!')
    exit()

#choose a set of stars simialr in flux or brighter than the target (minflux)
median_peak=np.median(peak_flux,axis=0)
ind_good=np.where((median_peak>50)*(median_peak<1e4))[0]

#choose fixed aperture or scaled aperture to do following calibration
flux_exp=np.copy(flux_fixed)
apr_rad=np.copy(apr_rad_fixed)
print('fixed aperture')

#no local sky subtraction: 0; with local sky subtraction: 2
flux_ind = 2
#axis=0, stars; axis=1, aperture
#normalize by median brightness of different nods: A nod by A nod, B nod by B nod
medd_flux_exp=np.zeros((flux_exp.shape[0],flux_exp.shape[1],flux_exp.shape[2]))
ind_Anod=files_info['nods'].data== 0
ind_Bnod=files_info['nods'].data== 1
for i in range(flux_fixed.shape[1]):
    medflux_A=np.median(flux_exp[ind_Anod,i,:, flux_ind],axis=0)
    medflux_B=np.median(flux_exp[ind_Bnod,i,:, flux_ind],axis=0)
    for j in range(flux_exp.shape[2]):
        medd_flux_exp[ind_Anod,i,j]=flux_exp[ind_Anod,i,j, flux_ind]/medflux_A[j]
        medd_flux_exp[ind_Bnod,i,j]=flux_exp[ind_Bnod,i,j, flux_ind]/medflux_B[j]

        #correct baseline of different nods
        num = np.min([np.sum(ind_Anod),np.sum(ind_Bnod)])
        shift1=np.median(medd_flux_exp[ind_Bnod, i, j][0:num]-medd_flux_exp[ind_Anod,i,j][0:num])
        medd_flux_exp[ind_Anod,i,j]+=shift1

max_iter=5
std_cut=3
slope_cut=3
final_curve=[]
final_refstar=[]
curve_targ=np.zeros((flux_exp.shape[0], apr_rad.size))
#bad star indices
bad_stars = np.array([])
#loop over aperture radius
for ap in range(apr_rad.size):
    #can fix the reference stars down after selecting out
    # ind_refstar=np.array([19, 20, 29, 44, 58])
    # cal_curves=[]
    # #first element is the calibrated light curve of the target
    # curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, ind_targ[0])
    # cal_curves.append(curve)
    # for i_star in ind_refstar:
    #     curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, i_star)
    #     cal_curves.append(curve)

    print('starting iteration of aperture radius: ', apr_rad[ap])
    #exclude target star from good stars
    ind_refstar = np.copy(ind_good[ind_good!=ind_targ[0]])
    #delete bad stars
    if bad_stars.size > 0:
        ind_refstar = ind_refstar[~np.isin(ind_refstar, bad_stars)]
    cal_curves=[]
    #first element is the calibrated light curve of the target
    curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, ind_targ[0])
    cal_curves.append(curve)
    for i_star in ind_refstar:
        curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, i_star)
        cal_curves.append(curve)
    std_stars=np.zeros(ind_refstar.size+1)
    slope_stars=np.zeros(ind_refstar.size+1)
    #calculate robust std and slope of calibrated light curves
    for i in range(ind_refstar.size+1):
        std_stars[i]=robust_sigma(np.roll(cal_curves[i],-1)-cal_curves[i])/np.sqrt(2)
        slope_stars[i]=robust_linear_fit(x_jdh, cal_curves[i])
    print('reference stars index of iteration0: ', ind_refstar)
    for k in range(max_iter):
        #choose next reference stars
        ind_refstar_new=choose_ref_stars(std_stars, slope_stars, ind_refstar, std=std_cut, slope=slope_cut)
        print('reference stars index of iteration{:}: '.format(k+1), ind_refstar_new)
        if np.array_equal(ind_refstar_new, ind_refstar):
            #the reference stars do not change in two iterations: we have found the well-behave reference stars
            break
        elif len(ind_refstar_new)==0:
            print('Warning: cannot find refernce stars in this iteration and return previous iteration!')
            break
        elif len(ind_refstar_new)==1:
            print('Warning: only select out one reference star and return previous iteration!')
            break
        else:
            ind_refstar=np.copy(ind_refstar_new)

        #light curve calibration
        cal_curves=[]
        curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, ind_targ[0])
        cal_curves.append(curve)
        for i_star in ind_refstar:
            curve,_=curve_calibration(medd_flux_exp[:,:,ap], ind_refstar, i_star)
            cal_curves.append(curve)
        std_stars=np.zeros(ind_refstar.size+1)
        slope_stars=np.zeros(ind_refstar.size+1)
        #calculate robust std and slope of calibrated light curves
        for i in range(ind_refstar.size+1):
            std_stars[i]=robust_sigma(np.roll(cal_curves[i],-1)-cal_curves[i])/np.sqrt(2)
            slope_stars[i]=robust_linear_fit(x_jdh, cal_curves[i])

    #save the result from the last iteration
    final_refstar.append(ind_refstar)
    curve_targ[:,ap]=np.copy(cal_curves[0])
    #final detrended(calibrated) curves: axis=0, aperture; axis=1, target+reference stars
    final_curve.append(cal_curves)

np.save(in_dir+'/final_curves.npy', final_curve)
np.save(in_dir+'/ind_final_refstar.npy', final_refstar)
#find out which aperture has the smallest rms
rms_curve=np.zeros(curve_targ.shape[1])
for i in range(curve_targ.shape[1]):
    rms_curve[i]= robust_sigma(np.roll(curve_targ[:,i],-1)-curve_targ[:,i])/np.sqrt(2)
rms_min=np.min(rms_curve)
rms_min_ind=np.argmin(rms_curve)
print('min rms', rms_min)
print('aperture radius of min rms',apr_rad[rms_min_ind])

#bin every n_bin points
n_bin = 1
#aperture size index
apr_r = rms_min_ind
JD_binned=np.mean(x_jdh.reshape(-1, n_bin), axis=1)
#also calculate calibration curve of the target
curve_targ, cal_targ=curve_calibration(medd_flux_exp[:,:,apr_r], final_refstar[apr_r], ind_targ[0])
curve_targ_bin=np.mean(curve_targ.reshape(-1, n_bin), axis=1)
cal_targ_bin=np.mean(cal_targ.reshape(-1, n_bin), axis=1)
#raw light curve of the target
raw_targ_bin=np.mean(medd_flux_exp[:,ind_targ[0],apr_r].reshape(-1, n_bin), axis=1)
curve_stars_bin=block_reduce(final_curve[apr_r][1:], block_size=(1, n_bin), func=np.mean)

print('outlier frame index')
print(np.where(np.logical_or(curve_targ<0.85, curve_targ>1.15)))
#bad frame mask (obtained by the fisrt run of this script: outliers in the detrened light curve)
bad_mask=np.ones_like(JD_binned, dtype='bool')
# bad_frame=np.array([148, 149, 150])-1
# bad_mask[bad_frame]=False

rms = robust_sigma(np.roll(curve_targ_bin[bad_mask],-1)-curve_targ_bin[bad_mask])/np.sqrt(2)
print('rms of binned light curve: ', rms)
# ind_star=ind_good[apr_r]
# x_jd=files_info['JD'].data
# #light curve calibration
# curve_targ, calib_curve, JD_binned=curve_calibration(medd_flux_exp, ind_good, ind_targ[0], n_bin=n_bin,JD=x_jd)
# curve_refstar1,_,_=curve_calibration(medd_flux_exp, ind_good, ind_star[0], n_bin=n_bin,JD=x_jd)
# curve_refstar2,_,_=curve_calibration(medd_flux_exp, ind_good, ind_star[1], n_bin=n_bin,JD=x_jd)
# #curve_refstar3,_,_=curve_calibration(medd_flux_fixed, ind_good, ind_star[3], n_bin=n_bin,JD=x_jd)
# #get the light curve of all reference stars for aperture index: apr_r
# curve_refstar=[]
# for i in range (len(ind_good[apr_r])):
#     light_curve, _, _=curve_calibration(medd_flux_exp, ind_good, ind_good[apr_r][i], n_bin=n_bin, JD=x_jd, aper_ind=apr_r)
#     curve_refstar.append(light_curve)
# curve_refstar=np.array(curve_refstar)

print('calibration reference star index', final_refstar[apr_r])
#also show the final reference stars in the images
im_ind=0
image=fits.getdata(path[im_ind],header=False)
positions=np.transpose((xy_found[im_ind, final_refstar[apr_r],0], xy_found[im_ind, final_refstar[apr_r],1]))
apertures=CircularAperture(positions, r=8.)
position_t=np.transpose((xy_found[im_ind, ind_targ[0],0], xy_found[im_ind, ind_targ[0],1]))
apertures_t=CircularAperture(position_t, r=8.)
norm=ImageNormalize(stretch=SqrtStretch())
plt.plot(size=(7,5))
plt.imshow(image,norm=norm,vmax=200,vmin=-200)
apertures.plot(color='red', lw=1.5, alpha=0.5)
apertures_t.plot(color='black', lw=1.5, alpha=0.5)
for i in range(len(final_refstar[apr_r])):
    plt.text(xy_found[im_ind, final_refstar[apr_r][i],0]+5, xy_found[im_ind, final_refstar[apr_r][i],1]+5, '{:}'.format(final_refstar[apr_r][i]), fontsize=15)
plt.text(xy_found[im_ind, ind_targ[0], 0]+5, xy_found[im_ind, ind_targ[0],1]+5, 'target', fontsize=15)
plt.xlabel('X [pixel]')
plt.ylabel('Y [pixel]')
plt.minorticks_on()
#check source detection
plt.tight_layout()
plt.savefig(in_dir+'/plots/reference_stars_aperturerad{:}.png'.format(apr_rad[apr_r]), dpi=200)
plt.close()

#make a plot of light curves
ax1=plt.subplot(311)
plt.plot(JD_binned[bad_mask], curve_targ_bin[bad_mask],'o')
plt.errorbar(JD_binned[3],curve_targ_bin[3], yerr=rms, capsize=2)
plt.hlines(1,xmin=0,xmax=10,colors='black',linestyles='solid')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.xlim(0, JD_binned.max())
plt.ylim(0.85,1.15)
plt.ylabel('Relative flux')
ax1.minorticks_on()
ax1.set_title('Detrended light curve '+target_name)

ax2=plt.subplot(312, sharex=ax1)
plt.plot(JD_binned[bad_mask],  cal_targ_bin[bad_mask],'o')
plt.hlines(1,xmin=0,xmax=10,colors='black',linestyles='solid')
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_title('Calibration light curve')
ax2.set_ylim(0.7, 1.3)
ax2.minorticks_on()

ax3=plt.subplot(313, sharex=ax1)
plt.plot(JD_binned[bad_mask], raw_targ_bin[bad_mask],'o')
plt.hlines(1,xmin=0,xmax=10,colors='black',linestyles='solid')
ax3.set_title('Raw light curve')
ax3.set_ylim(0.7, 1.3)
ax3.minorticks_on()

plt.xlabel('Elapsed time [hr]')
plt.tight_layout()
plt.savefig(in_dir+'/plots/light_curves_bin{:}_aper{:}.png'.format(n_bin,apr_rad[apr_r]), dpi=200)
plt.close()

#plot out the detrended light curves of referece stars
n_plots=len(final_refstar[apr_r])
fig, axs = plt.subplots(n_plots, 1, figsize=(12,16), sharex=True, sharey=True, constrained_layout=True)
for i in range(n_plots):
    axs[i].plot(JD_binned, curve_stars_bin[i],'o')
    axs[i].hlines(1,xmin=0,xmax=10,colors='black',linestyles='solid')
    axs[i].text(0.1, 1.1, '{:}'.format(final_refstar[apr_r][i]), fontsize=15)
axs[0].set_xlim(0, JD_binned.max())
axs[0].set_ylim(0.85, 1.15)
axs[-1].set_xlabel('Elapsed time [hr]')
axs[0].set_ylabel('Relative flux')
axs[0].minorticks_on()
axs[0].set_title('Detrened light curve of reference stars')
plt.savefig(in_dir+'/plots/ref_light_curves_bin{:}_aper{:}.png'.format(n_bin,apr_rad[apr_r]), dpi=200)
plt.close()

#also plot the raw light curves of reference stars
raw_stars_bin=block_reduce(medd_flux_exp[:,final_refstar[apr_r],apr_r], block_size=(n_bin, 1), func=np.mean)
n_plots=len(final_refstar[apr_r])
fig, axs = plt.subplots(n_plots, 1, figsize=(12,16), sharex=True, sharey=True, constrained_layout=True)
for i in range(n_plots):
    axs[i].plot(JD_binned, raw_stars_bin[:,i],'o')
    axs[i].hlines(1, xmin=0, xmax=10,colors='black',linestyles='solid')
    axs[i].text(0.1, 1.1, '{:}'.format(final_refstar[apr_r][i]), fontsize=15)
axs[0].set_xlim(0, JD_binned.max())
axs[0].set_ylim(0.6, 1.4)
axs[-1].set_xlabel('Elapsed time [hr]')
axs[0].set_ylabel('Relative flux')
axs[0].minorticks_on()
axs[0].set_title('Raw light curve of reference stars')
plt.savefig(in_dir+'/plots/refraw_light_curves_bin{:}_aper{:}.png'.format(n_bin,apr_rad[apr_r], std_cut, slope_cut), dpi=200)
plt.close()


#plot fitted xy positions in frames
plt.figure()
plt.plot(xy_found[:,ind_targ[0],0],xy_found[:,ind_targ[0],1],'o')
plt.xlabel('X [pixel]')
plt.ylabel('Y [pixel]')
plt.savefig(in_dir+'/plots/xy_fitted_positions.png', dpi=200)
plt.close()
#print(np.where(xy_found[:,ind_targ[0],0]>425))

#plot relative change of fitted xy positions in frames
y_drift=np.zeros(xy_found.shape[0])
x_drift=np.zeros(xy_found.shape[0])
total_drift=np.zeros_like(y_drift)
firstAind=np.argwhere(ind_Anod==True).reshape(-1)[0]
firstBind=np.argwhere(ind_Bnod==True).reshape(-1)[0]
#correct baseline in y
y_drift[ind_Anod]=xy_found[ind_Anod,ind_targ[0],1]-xy_found[firstAind,ind_targ[0],1]
y_drift[ind_Bnod]=xy_found[ind_Bnod,ind_targ[0],1]-xy_found[firstBind,ind_targ[0],1]
x_drift[ind_Anod]=xy_found[ind_Anod,ind_targ[0],0]-xy_found[firstAind,ind_targ[0],0]
x_drift[ind_Bnod]=xy_found[ind_Bnod,ind_targ[0],0]-xy_found[firstBind,ind_targ[0],0]
total_drift=np.sqrt(x_drift**2+y_drift**2)
#plot x against y
plt.figure()
plt.plot(x_drift[ind_Anod], y_drift[ind_Anod],'-o',label='A nod')
plt.plot(x_drift[ind_Bnod], y_drift[ind_Bnod],'-o', label='B nod')
plt.ylabel('Y drift [pixel]')
plt.xlabel('X drift [pixel]')
plt.legend()
plt.minorticks_on()
plt.tight_layout()
plt.savefig(in_dir+'/plots/xy_reldrift.png', dpi=200)
plt.close()

#
# #fit low-order polynomial to the detrended light curve
# x_jd=files_info['JD'].data
# x_jdh=(x_jd-x_jd[0])*24
# fitted_curve=np.zeros_like(curve_targ)
# std_rvfit=np.zeros(curve_targ.shape[1])
# rms_curve=np.zeros(curve_targ.shape[1])
# #loop over aperture size
# for i in range(curve_targ.shape[1]):
#     #z = np.polyfit(x_jdh, curve_targ[:,i],1)
#     #p = np.poly1d(z)
#     #fitted_curve[:,i]=p(x_jdh)
#     #std_rvfit[i]=np.std(curve_targ[:,i]-fitted_curve[:,i])
#     #std with low-pass filtering
#     #_, _, rms_curve[i]= sigma_clipped_stats(np.roll(curve_targ[:,i],-1)-curve_targ[:,i], sigma=3.0)/np.sqrt(2)
#     rms_curve[i]= robust_sigma(np.roll(curve_targ[:,i],-1)-curve_targ[:,i])/np.sqrt(2)
#
# #find which aperture radius has the smallest std
# # std_min=np.min(std_rvfit)
# # std_min_ind=np.argmin(std_rvfit)
# # print('min std', std_min)
# # print('aperture radius',apr_rad[std_min_ind])
#
# rms_min=np.min(rms_curve)
# rms_min_ind=np.argmin(rms_curve)
# print('min rms', rms_min)

#arcsec per pixel
pixel_scale=0.288
#plot airmass and seeings
fig, axs = plt.subplots(4,1,figsize=(10,15),sharex=True, constrained_layout=True)
axs[0].plot(x_jdh, files_info['airmass'].data,'-o')
#plt.setp(ax1.get_xticklabels(), visible=False)
axs[0].set_ylabel('Airmass')
axs[0].minorticks_on()
axs[1].plot(x_jdh, files_info['dimm_fwhm'].data,'o')
axs[1].set_ylabel('Seeing [arcsec]')
axs[1].minorticks_on()
#measured seeing
axs[2].plot(x_jdh, pixel_scale*files_info['median_fwhm'].data,'o')
axs[2].set_ylabel('Median FWHM [arcsec]')
axs[2].minorticks_on()
#position drift
axs[3].plot(x_jdh[ind_Anod], total_drift[ind_Anod],'o', label='A nod')
axs[3].plot(x_jdh[ind_Bnod], total_drift[ind_Bnod],'o', label='B nod')
axs[3].set_xlabel('Elapsed time [hr]')
axs[3].set_ylabel('Drift [pixel]')
axs[3].minorticks_on()
plt.legend()
plt.savefig(in_dir+'/plots/airmass_seeing.png', dpi=200)
plt.close()

np.savetxt(in_dir+'/median_fwhm.txt', pixel_scale*files_info['median_fwhm'].data)

# if np.nansum(flux_exp-flux_fixed)==0:
#     print('aperture radius of min rms',apr_rad[rms_min_ind])
# else:
#     print('aperture radius of min rms [fwhm]',apr_rad_scaled[rms_min_ind])
print('median fwhm of all frames median_fwhm', np.median(files_info['median_fwhm']))
print('Elapsed time (h):', x_jdh[-1]-x_jdh[0])

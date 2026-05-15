#photometry measurement of data after sky subtraction
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.modeling import models, fitting
import glob
from photutils.detection import DAOStarFinder, IRAFStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.nddata import Cutout2D
from astropy.table import QTable
from astropy.nddata import block_reduce
from photutils.aperture import CircularAperture,aperture_photometry,CircularAnnulus
from photutils.centroids import centroid_sources,centroid_1dg,centroid_2dg
from scipy.signal import correlate2d
from tqdm import tqdm
from lightcurve_reduction.special_functions import aper_sub_sky, check_position
import os, errno

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (8, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

#identify nod: the number of unique total offsets values
def identify_nods(offset):
    tot_offset=np.sqrt(offset[:,0]**2+offset[:,1]**2)
    unique_offset=sorted(np.unique(tot_offset))
    nods=np.zeros(offset.shape[0])
    for i in range(len(unique_offset)):
        ind= tot_offset==unique_offset[i]
        nods[ind]=i
    return nods

def pos_ref_shift(files, x_centroid, y_centroid, peak, offset,bad_pixel):
    #choose some bright star(s) near the center of the detector
    #find the shift of it by fitting its centriods in each frame
    x_maxoff=max(abs(offset[:,0]))
    y_maxoff=max(abs(offset[:,1]))
    #the area should be adjusted for different dataset
    ind_center=(x_centroid>250)*(x_centroid<750)*(y_centroid>250)*(y_centroid<750)
    #ind_center=(x_centroid>100)*(x_centroid<500)*(y_centroid>200)*(y_centroid<750)
    ind_peak=(peak<8000)*(peak>1000)
    ind=ind_center*ind_peak
    #select stars that meet above conditions
    x_ref=x_centroid[ind]
    y_ref=y_centroid[ind]
    x_fitted=np.zeros((files.shape[0],x_ref.size))
    y_fitted=np.zeros_like(x_fitted)
    x_fitted[0]=np.copy(x_ref)
    y_fitted[0]=np.copy(y_ref)
    #find centroid, skip the first frame
    for i in range(1,files.shape[0]):
        x_ref_guess=x_ref+offset[i,0]
        y_ref_guess=y_ref+offset[i,1]
        #fitting 2D Gaussians (2dg) or 1d Gaussian magrinal x,y distribution (1dg)
        x_fitted[i,:], y_fitted[i,:]=centroid_sources(files[i], x_ref_guess, y_ref_guess, box_size=13,
                                                      mask=bad_pixel, centroid_func=centroid_1dg)

    offset_fitted=np.zeros_like(offset)
    #fitted offset averaging selected bright stars
    offset_fitted[:,0]=np.mean(x_fitted-x_ref,axis=1)
    offset_fitted[:,1]=np.mean(y_fitted-y_ref,axis=1)
    #also return bright star's index
    return offset_fitted,ind


def median_field_fwhm(data,x,y):
    #calculate median fwhm for an image and list of stars xy
    fwhm=np.zeros(x.size,2)
    for i in range(x.size):
        if x[i]>0 and x[i]<1023 and y[i]>0 and y[i]<1023:
            #2d Gaussian fitting
            box_size=11
            cutout=Cutout2D(data,(x[i],y[i]),box_size)
            xx, yy = np.meshgrid(np.arange(box_size), np.arange(box_size))
            yx_position_cutout=[box_size//2,box_size//2]
            g=models.Gaussian2D(amplitude=cutout.data.max(),
                                x_mean=yx_position_cutout[1],
                                y_mean=yx_position_cutout[0],
                                x_stddev=2,
                                y_stddev=2)
            fitter=fitting.LevMarLSQFitter()
            par = fitter(g, xx, yy, cutout.data)
            fwhm[i,0]=par.x_fwhm
            fwhm[i,1]=par.y_fwhm
    #why compare with 0.7?
    ind=fwhm[:,0]>0.7
    #return np.median(fwhm[ind,:],axis=0)
    return np.median(fwhm[ind,:])

def position_pairs(x,y):
    #make position pairs
    position=[]
    for i in range(x.size):
        position.append((x[i],y[i]))
    return position

def add_stars(star_xyf, x_good, y_good, peak_good):
    #star_xyf: stars to manually add into reference stars
    x_good = np.append(x_good, star_xyf[:, 0])
    y_good = np.append(y_good, star_xyf[:, 1])
    peak_good = np.append(peak_good, star_xyf[:, 2])
    return x_good, y_good, peak_good


def delete_stars(star_xy, x_good, y_good, peak_good):
    #star_xy: stars to manually delete from reference stars
    rad = 15
    for i in range(star_xy.shape[0]):
        ind=np.where((x_good > (star_xy[i, 0] - rad))*(x_good < star_xy[i, 0] + rad)*\
                     (y_good > (star_xy[i, 1] - rad))*(y_good < star_xy[i, 1] + rad))[0]
        if ind == 0:
            continue
        else:
            x_good = np.delete(x_good, ind)
            y_good = np.delete(y_good, ind)
            peak_good = np.delete(peak_good, ind)
    return x_good, y_good, peak_good

def Gaussian_2d(data,x,y,box_size=11):
    xy_fwhm_fitted=np.zeros(2)
    xy_fitted=np.zeros(2)
    #2d Gaussian fitting
    cutout=Cutout2D(data,(x,y),box_size)
    xx, yy = np.meshgrid(np.arange(box_size), np.arange(box_size))
    yx_position_cutout=[box_size//2,box_size//2]
    g=models.Gaussian2D(amplitude=cutout.data.max(),
                        x_mean=yx_position_cutout[1],
                        y_mean=yx_position_cutout[0],
                        x_stddev=1,
                        y_stddev=1,
                        bounds={'x_mean':(-box_size*0.5,box_size*0.5),'y_mean':(-box_size*0.5,box_size*0.5)})
    fitter=fitting.LevMarLSQFitter()
    par = fitter(g, xx, yy, cutout.data)
    xy_fit_position_orig = cutout.to_original_position((par.x_mean.value, par.y_mean.value))
    xy_fwhm_fitted[0]=par.x_fwhm
    xy_fwhm_fitted[1]=par.y_fwhm
    xy_fitted[0]=xy_fit_position_orig[0]
    xy_fitted[1]=xy_fit_position_orig[1]
    return xy_fitted, xy_fwhm_fitted

def r_theta(im, xc, yc):
    # returns the radius rr and the angle phi for point (xc,yc)
    ny, nx = im.shape
    yp, xp = np.mgrid[0:ny,0:nx]
    yp = yp - yc
    xp = xp - xc
    rr = np.sqrt(np.power(yp,2.) + np.power(xp,2.))
    phi = np.arctan2(yp, xp)
    return(rr, phi)



#read in files
target_name='J0819-0335'
date='2022_02_10'
in_dir='/Users/data/SOFI/'+date+'/'+target_name+'/skysub'
out_dir='/Users/data/SOFI/'+date+'/'+target_name+'/reduced'
path=sorted(glob.glob(in_dir+'/*skysub.fits'))
n_files=len(path)
files=np.empty((n_files,1024,1024),'float64')
JD=np.empty(n_files,'float64')
airmass=np.empty(n_files,'float64')
dimm_fwhm=np.empty(n_files,'float64')
files_offset=np.empty((n_files,4),'float64')

plot_dir = out_dir + '/plots'
try:
    os.mkdir(plot_dir)
    print('Directory' + plot_dir + ' created successfully')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for i in range(n_files):
    files[i],header=fits.getdata(path[i],header=True)
    JD[i]=header['MJD-OBS']
    airmass[i]=header['HIERARCH ESO TEL AIRM START']
    dimm_fwhm[i]=header['HIERARCH ESO TEL AMBI FWHM START']
    # files_offset[i,0]=header['HIERARCH ESO SEQ RELOFFSETX']
    # files_offset[i,1]=header['HIERARCH ESO SEQ RELOFFSETY']
    # files_offset[i,2]=header['HIERARCH ESO SEQ CUMOFFSETX']
    # files_offset[i,3]=header['HIERARCH ESO SEQ CUMOFFSETY']

#calculate relative offsets to absolute offsets
#offset=np.zeros((n_files,2),'float64')
#first step of first template
#
# offset[0,0]=files_offset[0,0]
# offset[0,1]=files_offset[0,1]
# for i in range(1,n_files):
#     #x
#     offset[i][0]=offset[i-1][0]+files_offset[i][0]
#     #y
#     offset[i][1]=offset[i-1][1]+files_offset[i][1]
#load in offset produced in skysub_nod.py after correction
offset=np.load(out_dir+'/offsets.npy')

#bin frames if needed for faint targets
# n_bin=3
# files=block_reduce(files, block_size=(n_bin,1,1),func=np.mean)
# JD=block_reduce(JD, block_size=n_bin,func=np.median)
# airmass=block_reduce(airmass, block_size=n_bin,func=np.median)
# dimm_fwhm=block_reduce(dimm_fwhm, block_size=n_bin,func=np.median)
# offset=block_reduce(offset, block_size=(n_bin,1),func=np.median)
# print('Frames are binned')
# n_files=files.shape[0]
# print(offset)
#exit()

#write out data in a single fits so that we can visually check each frame, no matter binned or not
bin_path=in_dir.replace('skysub','binned_frames.fits')
fits.writeto(bin_path,files,overwrite=True)

#consider to use the bad pixel map provided by eso
bad_pixel=np.isnan(files[0])
#show bad pixel map
plt.plot(size=(7,5))
plt.imshow(bad_pixel)
plt.savefig(out_dir+"/plots/bad_pixel.png")
plt.close()


#source dection
mean, median, std = sigma_clipped_stats(files[0], mask=bad_pixel, sigma=3.0)
daofind = DAOStarFinder(fwhm=5.0, threshold=5.*std, sharplo=0.2, sharphi=0.65, roundlo=-1.0, roundhi=1.0)
#daofind = DAOStarFinder(fwhm=5.0, threshold=5.*std, sharplo=0.05, sharphi=0.65, roundlo=-1.0, roundhi=1.0)
sources = daofind(files[0] - median)
#print(sources)

#visualize source detecion
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.plot(size=(7,5))
plt.imshow(files[0],norm=norm,vmax=200,vmin=-200)
apertures.plot(color='red', lw=1.5, alpha=0.5)
#check source detection
plt.savefig(plot_dir + '/sources_detection.png')
plt.close()

#initialize a group of reference stars
good_ind=(sources['peak']>20)*(sources['peak']<1e4)
#do not choose stars near the edge
ind_range=(sources['xcentroid']>20)*(sources['xcentroid']<1000)*(sources['ycentroid']>20)*(sources['ycentroid']<930)
#ind_range=(sources['xcentroid']>70)*(sources['xcentroid']<850)*(sources['ycentroid']>200)*(sources['ycentroid']<950)
good_ind*=ind_range

#find the offset from the fitting of bright stars
x_good=sources['xcentroid'][good_ind].data
y_good=sources['ycentroid'][good_ind].data
#note: peak is subtracted by bkg median in sources
peak_good=sources['peak'][good_ind].data

#add or delete reference stars manually here
# add_stars_xyf = np.array([])
# x_good, y_good, peak_good = add_stars(add_stars_xyf, x_good, y_good, peak_good)
#remove stars
# del_star_xy = np.array([])
# x_good, y_good, peak_good = delete_stars(del_star_xy, x_good, y_good, peak_good)

#rough offset found by fiting the location of bright stars: suitable for images that do not drift a lot
offset_fitted, bright_ind=pos_ref_shift(files,x_good, y_good, peak_good, offset, bad_pixel)
print('bright_ind', bright_ind)

#visualize reference stars
positions = np.transpose((x_good, y_good))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.plot(size=(7,5))
plt.imshow(files[0],norm=norm,vmax=200,vmin=-200)
apertures.plot(color='red', lw=1.5, alpha=0.5)
#check source detection
plt.savefig(out_dir+"/plots/reference_stars.png")
plt.close()

# print(sources)
# exit()

#identify nods
nods=identify_nods(offset)
#cross correlation method to find the drift of images
#nods=0, template: files[0]; nods=1, template: file[3]
template1=np.copy(files[0][400:601,400:601])
template1-=template1.mean()
template2=np.copy(files[3][400:601,400:601])
template2-=template2.mean()
shift=np.copy(offset)
xy_fwhm=np.zeros((n_files,x_good.size,2))
xy_found=np.zeros((n_files,x_good.size,2))

#check if found the offset is correct
positions = np.transpose((x_good+offset[3,0], y_good+offset[3,1]))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.plot(size=(7,5))
plt.imshow(files[3],norm=norm,vmax=200,vmin=-200)
apertures.plot(color='red', lw=1.5, alpha=0.5)
#check source detection
plt.savefig(out_dir+"/plots/check_offset.png")
plt.close()

print('fit the location of good stars in frames')
for i in tqdm(range(n_files)):
    #first method: use cross-correlation to find the approximate offsets (pixel)
    #It is slower but can deal with dataset that have large drifts.
    #The founded drift still may be wrong in some frames, and we should check the outliers in the final light curve
    # data=np.copy(files[i][400:601,400:601])
    # data-=data.mean()
    # if nods[i]==0:
    #     corr=correlate2d(template1,data,boundary='fill', mode='same')
    #     corr_y, corr_x = np.unravel_index(np.argmax(corr), corr.shape)
    #     #100 is calculated from 400:600
    #     shift[i,0]=100-corr_x
    #     shift[i,1]=100-corr_y
    #     shift[i]+=offset[0]
    # elif nods[i]==1:
    #     corr=correlate2d(template2,data,boundary='fill', mode='same')
    #     corr_y, corr_x = np.unravel_index(np.argmax(corr), corr.shape)
    #     shift[i,0]=100-corr_x
    #     shift[i,1]=100-corr_y
    #     shift[i]+=offset[3]
    #
    # x_shift=x_good+shift[i,0]
    # y_shift=y_good+shift[i,1]
    # for j in range(x_shift.size):
    #     if x_shift[j]>10 and x_shift[j]<1010 and y_shift[j]>10 and y_shift[j]<1010:
    #         xy_found[i,j],xy_fwhm[i,j]=Gaussian_2d(files[i],x_shift[j],y_shift[j],box_size=11)

    #second method: faster, applicable to images that do not drift a lot
    x_shift, y_shift=centroid_sources(files[i], x_good+offset[i,0],
                                                 y_good+offset[i,1], box_size=15,
                                                 mask=bad_pixel,
                                                 centroid_func=centroid_1dg)
    for j in range(x_shift.size):
        if x_shift[j]>10 and x_shift[j]<1010 and y_shift[j]>10 and y_shift[j]<1010:
            xy_found[i,j],xy_fwhm[i,j]=Gaussian_2d(files[i],x_shift[j],y_shift[j],box_size=13)

    #third method: only use Guassian 2d fitting: only useful if the image drift only a few pixels
    # for j in range(x_good.size):
    #     xy_found[i,j],xy_fwhm[i,j]=Gaussian_2d(files[i],x_good[j]+offset[i,0],y_good[j]+offset[i,1],box_size=13)


#print(xy_found[:,1])
#check the fitted position
#fit_positions=position_pairs(xy_found[150,:,0],xy_found[150,:,1])
positions = np.transpose((xy_found[34,:,0],xy_found[34,:,1]))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())
plt.plot(size=(7,5))
plt.imshow(files[34],norm=norm,vmax=200,vmin=-200)
apertures.plot(color='red', lw=1.5, alpha=0.5)
#check source detection
plt.savefig(plot_dir + '/check_fit_position.png')
plt.close()


#check the shift read from headers or found from cross correlation
plt.figure()
plt.plot(xy_found[:,0,0],xy_found[:,0,1],'o')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.savefig(plot_dir + '/fitted_position.png')
plt.close()

#plot the fitting position of a single star
plt.figure()
plt.plot(shift[:,0],shift[:,1],'o')
plt.xlabel('shift x')
plt.ylabel('shift y')
plt.savefig(plot_dir + '/found_shift.png')
plt.close()

#check fitted positions: make a gif of all images
print('Checking fitted positions')
check_position(files, out_dir, xy_found)

#aperture photometry
apr_rad=np.array([3.0, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7.0])
apr_rad_scaled=np.array([0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5])
median_fwhm=np.zeros(n_files)
flux_fixed=np.zeros((n_files,x_good.size,apr_rad.size,3))
flux_scaled=np.zeros((n_files,x_good.size,apr_rad_scaled.size,3))
peak_flux=np.zeros((n_files,x_good.size))
print('do aperture photometry')
for i in tqdm(range(n_files)):
    #fixed aperture
    fwdm_ind=np.where(xy_fwhm[1,:]>0.7)
    #median fwhm of the frame
    median_fwhm[i]=np.median(xy_fwhm[i,fwdm_ind[0],fwdm_ind[1]])
    positions=position_pairs(xy_found[i,:,0],xy_found[i,:,1])
    sky_rad=[18,28]
    for j in range(apr_rad.size):
        #aperture photometry for fixed aperture
        phot=aper_sub_sky(files[i],positions,apr_rad[j],sky_rad,bad_pixel)
        flux_fixed[i,:,j,0]=phot['aperture_sum'].data
        flux_fixed[i,:,j,1]=phot['aper_bkg'].data
        flux_fixed[i,:,j,2]=phot['aper_sum_bkgsub'].data
        #aperture photometry for varying apertures
    sky_rad_scaled=np.array([4,6])*median_fwhm[i]
    for j in range(apr_rad_scaled.size):
        phot=aper_sub_sky(files[i],positions,apr_rad_scaled[j]*median_fwhm[i], sky_rad_scaled, bad_pixel)
        flux_scaled[i,:,j,0]=phot['aperture_sum'].data
        flux_scaled[i,:,j,1]=phot['aper_bkg'].data
        flux_scaled[i,:,j,2]=phot['aper_sum_bkgsub'].data
    #find peak pixel value too
    for k in range(x_good.size):
        rr,_=r_theta(files[i], xy_found[i,k,0],xy_found[i,k,1])
        peak_flux[i,k]=np.nanmax(files[i][rr<5])

#save out annulus_data
np.save(out_dir+'/flux_scaled.npy',flux_scaled)
np.save(out_dir+'/flux_fixed.npy',flux_fixed)
np.save(out_dir+'/peak_flux.npy',peak_flux)
np.save(out_dir+'/xy_found.npy',xy_found)
np.save(out_dir+'/files_shift.npy',shift)


t = QTable([JD, airmass, dimm_fwhm, median_fwhm, nods],
           names=('JD', 'airmass', 'dimm_fwhm','median_fwhm', 'nods'),
          meta={'name': target_name})
ascii.write(t, out_dir+'/files_info.dat', overwrite=True)

#subtract sky background
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm
from lightcurve_reduction.special_functions import robust_sigma
from astropy.stats import sigma_clipped_stats
import os, errno

#find offset for each files
# def find_offsets(files_dir):
#     n_elm=len(files_dir)
#     offsets=np.zeros((n_elm,3))
#     reloff_x=np.zeros(n_elm)
#     reloff_y=np.zeros(n_elm)
#     JD=np.zeros(n_elm)
#     template_order=np.zeros(n_elm)
#
#     for i in range(n_elm):
#         header=fits.getheader(files_dir[i])
#         template_order[i]=header['HIERARCH ESO OBS TPLNO']
#         reloff_x[i]=header['HIERARCH ESO SEQ RELOFFSETX']
#         reloff_y[i]=header['HIERARCH ESO SEQ RELOFFSETY']
#         JD[i]=header['MJD-OBS']
#
#     offsets[0,0]=reloff_x[0]
#     offsets[0,1]=reloff_y[0]
#     for i in range(1,n_elm):
#         #absolute offset to the fisrt frame
#         offsets[i,0]=offsets[i-1,0]+reloff_x[i]
#         offsets[i,1]=offsets[i-1,1]+reloff_y[i]
#     offsets[:,2]=(JD-JD[0])*24
#     return offsets

def match_sky(offsets, i):
    dt=abs(offsets[i,2]-offsets[:,2])
    off_abs=np.sqrt(offsets[:,0]**2+offsets[:,1]**2)
    #old: find sky frames of different nod, less than 15 mins apart
    #ind_nod=np.argwhere((off_abs!=off_abs[i])&(dt<0.25)).reshape(-1)
    #new: find sky frames of different nod, single frame closet in time
    ind_nod=np.argwhere(off_abs!=off_abs[i]).reshape(-1)
    closest=np.argmin(dt[ind_nod])
    #sky_diff=abs(ind_nod-i)
    return ind_nod[closest]


#----------------------------------------------------
parent_dir='/Users/data/SOFI'
date='/2022_02_10'
tar_name='/J0819-0335'
in_dir=parent_dir + date + tar_name
sky_dir=in_dir + '/skysub'
red_dir=in_dir + '/reduced'
path=sorted(glob.glob(in_dir + '/xtalk/*.fits'))
n_files=len(path)
err_dir = in_dir + '/skysub_error'

try:
    os.mkdir(sky_dir)
    print('Directory ' + sky_dir + ' created successfully')
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Warning: ' + sky_dir + ' already exists')
    else:
        raise

try:
    os.mkdir(red_dir)
    print('Directory ' + red_dir + ' created successfully')
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Warning: ' + red_dir + ' already exists')
    else:
        raise

# try:
#     os.mkdir(err_dir)
#     print('Directory ' + err_dir + ' created successfully')
# except OSError as e:
#     if e.errno == errno.EEXIST:
#         print('Warning: ' + err_dir + ' already exists')
#     else:
#         raise

#gain: e/adu
g=5.4
#readout noise: 12e
sigma_readout = 12
sigma_readADU = 12/g
#correct flat field and illumination correction
#should change the dir of mflat and illum if thery are in a different folder
mflat=fits.getdata(parent_dir + date + '/mflat.fits', header=False)
illum=fits.getdata(parent_dir + date + '/illum.fits', header=False)
mflat/=np.median(mflat)
mflat*=illum
bad_pixel_map=fits.getdata(parent_dir + '/bad_pix_mask_october2012.fits', header=False).astype('bool')

#bad pixel flagging
_, med0, _=sigma_clipped_stats(mflat, sigma=3, maxiters=3)
rms=robust_sigma(mflat)
ind_bad=np.logical_or(abs(mflat-med0)>10*rms, bad_pixel_map)
#can we replace bad pixels?
mflat[ind_bad]=np.nan
#save updated bab pixel map
fits.writeto(in_dir + '/bad_pixel_map.fits', ind_bad*1, overwrite=True)

#read in all frames
frames=np.zeros((n_files,1024,1024))
err2_frames=np.zeros((n_files,1024,1024))
offsets=np.zeros((n_files,3))
reloff_x=np.zeros(n_files)
reloff_y=np.zeros(n_files)
JD=np.zeros(n_files)
hdu=[]
for i in range(n_files):
    frames[i],header=fits.getdata(path[i],header=True)
    reloff_x[i]=header['HIERARCH ESO SEQ RELOFFSETX']
    reloff_y[i]=header['HIERARCH ESO SEQ RELOFFSETY']
    JD[i]=header['MJD-OBS']
    hdu.append(header)
    #calculate photon error
    # err2_frames[i] = sigma_readADU**2 + frames[i]/g

#old: offsets=find_offsets(path)
#find spatial offsets + time for each file
offsets[0,0]=reloff_x[0]
offsets[0,1]=reloff_y[0]
for i in range(1,n_files):
    #absolute offset to the fisrt frame
    offsets[i,0]=offsets[i-1,0]+reloff_x[i]
    offsets[i,1]=offsets[i-1,1]+reloff_y[i]
offsets[:,2]=(JD-JD[0])*24

#special treatment for some target that has incorret offset
#offsets[198:,1]-=69.444444
print('offsets \n', offsets[:])
#print(path[108])
#print(path[90:97])
#print(np.where(offsets[:,1]>70))
#exit()
#discard bad frames for J0226 in Oct 22
# bad_frames=[i for i in range(90,97)]
# print(bad_frames)
# offsets=np.delete(offsets, bad_frames, axis=0)
# frames=np.delete(frames, bad_frames, axis=0)
# del hdu[90:97]
# del path[90:97]
# n_files=frames.shape[0]
#save out offsets
np.save(red_dir + '/offsets.npy',offsets[:,:2])

for i in tqdm(range(n_files)):
    #match frame to frames of different nods which are closest in time
    sky_ind=match_sky(offsets, i)
    # nsky=sky_ind.size
    # sky_im=np.copy(frames[sky_ind])
    # for j in range(nsky):
    #     _, med, _=sigma_clipped_stats(sky_im[j], sigma=3, maxiters=3,mask=ind_bad)
    #     #scale sky
    #     sky_im[j]/=med
    # #median sky
    # sky_med=np.median(sky_im,axis=0)
    #scale sky to the image
    _, med1, _=sigma_clipped_stats(frames[sky_ind], sigma=3, maxiters=3,mask=ind_bad)
    _, med2, _=sigma_clipped_stats(frames[i], sigma=3, maxiters=3,mask=ind_bad)
    scale_fac=med2/med1
    #subract sky to image
    frame_subsky=(frames[i]-frames[sky_ind]*scale_fac)/mflat
    new_path=path[i].replace(in_dir+'/xtalk', sky_dir).replace('xtalk', 'skysub')
    fits.writeto(new_path,frame_subsky, hdu[i],overwrite=True)
    # #also write out error for every pixel
    # np.seterr(invalid='ignore')
    # err_subsky = np.sqrt(err2_frames[i] + err2_frames[sky_ind]*(scale_fac**2))/mflat
    # err_path=path[i].replace(in_dir+'/xtalk', err_dir).replace('xtalk', 'skysub_err')
    # fits.writeto(err_path,err_subsky, hdu[i],overwrite=True)

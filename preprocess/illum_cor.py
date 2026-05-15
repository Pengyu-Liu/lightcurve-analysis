#illumination correction
import numpy as np
import glob
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.modeling import models, fitting
from photutils.centroids import centroid_1dg,centroid_2dg
from tqdm import tqdm
from lightcurve_reduction.special_functions import split_sets, xtalk, aper_sub_sky, surf2d, surf_polyfit2d
from photutils.aperture import CircularAperture,aperture_photometry,CircularAnnulus


#read in illumination correction files
illu_dir='/Users/data/SOFI/2022_02_13/std'
out_dir='/Users/data/SOFI/2022_02_13'
path=sorted(glob.glob(illu_dir+'/*.fits'))
n_files=len(path)

#extract time from headers
mjd=np.zeros(n_files)
filter=[]
for i in range(n_files):
    h=fits.getheader(path[i])
    mjd[i]=h['MJD-OBS']
    filter.append(h['HIERARCH ESO INS FILT1 NAME'])
#hours
t=24*(mjd-mjd[0])
#choose band
band ='Js'
ind_J=[i for i, e in enumerate(filter) if e == band]
t=t[ind_J]
mjd=mjd[ind_J]
filter_tmp=[]
path_tmp=[]
for i in ind_J:
    filter_tmp.append(filter[i])
    path_tmp.append(path[i])
filter=filter_tmp
path=path_tmp

#split sets separated by at least 5 hours
#illum grid were taken on multiple days
#we only need one(best) set of 16
#do it or not? I think probably we will only have 16 std frames and we can use them direclty
parts, nparts, starts, lens=split_sets(t,5)

fluxes=np.zeros((nparts, 16))
zsurf=np.zeros((nparts, 1024, 1024))

#Standard stars used were identified by name from ESO headers,
#and in the images using finder charts from SIMBAD/Aladin
#the approximate pixel positions of the STD are encoded in xpos0,ypos0
#xpos0=xpos[k]
#ypos0=ypos[k]
run_no=1
if run_no == 1:
   #for PSO318 dataset: 107, 71
   #for PSO071: 88, 87
   #for Oct 21: 75, 116
   #for Oct 22: 77, 109
   xpos0=77
   ypos0=109
else:
   xpos0=88
   ypos0=87

#read in flat field
mflat=fits.getdata(out_dir+'/mflat_' + band + '.fits',header=False)
mflat/=np.median(mflat)

#choose which set to use
for k in range(nparts):
    ind=np.argwhere(parts==k).reshape(-1)
    nf=ind.size
    grid0=np.zeros((nf, 1024, 1024))
    for i in range(nf):
        grid0[i]=fits.getdata(path[ind[i]],header=False)
         #remove xtalk for each file
        grid0[i]=xtalk(grid0[i], aplpha=1.4e-5)
    grid=np.zeros_like(grid0)
    #subtract sky from grid and flat field
    for i in range(nf-1):
        grid[i]=(grid0[i]-grid0[i+1])/mflat
    grid[-1]=(grid0[-1]-grid0[-2])/mflat

    #grid dither pattern?
    dim1=4
    dim2=4
    if run_no ==1:
       yoffs=[0,-5,-5,4,0]
       xoffs=[0,0,-5,0,-6]
    else:
       yoffs=[0,-5,-5,0,-5,0]
       xoffs=[0,2,0,7,8,5,0]

   # xx=reform(fltarr(nf),4,4)
   # yy=reform(fltarr(nf),4,4)
    xx=np.zeros(nf).reshape(4,4)
    yy=np.zeros(nf).reshape(4,4)
    for d1 in range(dim1):
       for d2 in range(dim2):
           yy[d1,d2]=ypos0+d2*260.+yoffs[k]
           xx[d1,d2]=xpos0+(d1*260.)*((d2 % 2) == 0) + (3*260.-d1*260)*((d2 % 2) != 0)+xoffs[k]

    xx=xx.transpose().reshape(-1)
    yy=yy.transpose().reshape(-1)
    print('xx',xx)
    print('yy',yy)
    #refine positions and measure flux
    #To-do: centroiding for Run-2 on out-of-focus stars could be improved,
    #but in practice this doesn't matter much since we are using a large photometry aperture
    flux=np.zeros(nf)
    for i in range(nf):
        x0=xx[i]
        y0=yy[i]
        im=grid[i]
        if run_no ==1:
            box=21
            cutout=Cutout2D(im,(x0,y0),box)
            subim= cutout.data
            xcntrd, ycntrd = centroid_1dg(subim)
            xy_fit_position_orig = cutout.to_original_position((xcntrd,  ycntrd))
            xtemp1= xy_fit_position_orig[0]
            ytemp1= xy_fit_position_orig[1]
        else:
            box=25
            cutout=Cutout2D(im,(x0,y0),box,copy=True)
            subim= cutout.data
            #or transpose?
            xxx, yyy = np.meshgrid(np.arange(box), np.arange(box))
            subim-=np.median(subim)
            xtemp1=round(x0)-box//2+np.sum(xxx*subim)/np.sum(subim)
            ytemp1=round(y0)-box//2+np.sum(yyy*subim)/np.sum(subim)

        if run_no ==1:
            #use 2d Guassian to fit
            box=21
            cutout=Cutout2D(im,(round(xtemp1),round(ytemp1)),box)
            subim=cutout.data
            x1, y1 = np.meshgrid(np.arange(box), np.arange(box))
            g0=models.Gaussian2D(amplitude=subim.max(),
                                x_mean=box//2,
                                y_mean=box//2,
                                x_stddev=3,
                                y_stddev=3)
            fitter=fitting.LevMarLSQFitter()
            par0 = fitter(g0, x1, y1, subim)
            xy_fit_position_orig = cutout.to_original_position((par0.x_mean.value, par0.y_mean.value))
            xtmp=xy_fit_position_orig[0]
            ytmp=xy_fit_position_orig[1]

            #fit again using par to get a better model
            box=11
            cutout=Cutout2D(im,(round(xtmp),round(ytmp)),box)
            x1, y1 = np.meshgrid(np.arange(box), np.arange(box))
            subim = cutout.data
            g0=models.Gaussian2D(amplitude=par0.amplitude.value,
                                x_mean=box//2,
                                y_mean=box//2,
                                x_stddev=par0.x_stddev.value,
                                y_stddev=par0.y_stddev.value,
                                theta=par0.theta.value)
            fitter=fitting.LevMarLSQFitter()
            par = fitter(g0, x1, y1, subim)
            xy_fit_position_orig = cutout.to_original_position((par.x_mean.value, par.y_mean.value))
            xx[i]=xy_fit_position_orig[0]
            yy[i]=xy_fit_position_orig[1]
        else:
            xx[i]=xtemp1
            yy[i]=ytemp1

        #some large aperture
        aper_rad=15
        skyrad=[21,31]
        #determine the flux and error for photometry radius 15
        phot=aper_sub_sky(im,([xx[i]-0.5,yy[i]-0.5],),aper_rad,skyrad,bad_pixel=None)
        flux[i]=phot['aper_sum_bkgsub'].data

    #2d polynomial fit (3 order) to the surface
    coeff, _, _, _=surf_polyfit2d(xx, yy, flux/np.median(flux))
    xxx,yyy=np.meshgrid(np.arange(1024),np.arange(1024))
    fitted_surf = surf2d(xxx, yyy, coeff)
    fluxes[k]=np.copy(flux)
    #2D surface fit to flux grid to estimate illumination correction
    zsurf[k]=np.copy(fitted_surf)

#choose one set
st=0
fits.writeto(out_dir+'/illum' + band + '.fits',zsurf[st],overwrite=True)

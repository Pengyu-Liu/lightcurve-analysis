#reduce flat field
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm
from lightcurve_reduction.special_functions import xtalk

in_dir='/Users/data/SOFI/2023_05_09/flatJ10'
out_dir='/Users/data/SOFI/2023_05_09/'
path=sorted(glob.glob(in_dir+'/*.fits'))
n_files=len(path)
frames=np.zeros((n_files, 1024, 1024))

#extract filter from headers
filter=[]
for i in range(n_files):
    h=fits.getheader(path[i])
    filter.append(h['HIERARCH ESO INS FILT1 NAME'])
#choose band
band = 'Js'
ind_J=[i for i, e in enumerate(filter) if e == band]
filter_tmp=[]
path_tmp=[]
for i in ind_J:
    filter_tmp.append(filter[i])
    path_tmp.append(path[i])
filter=filter_tmp
path=path_tmp
n_files=len(path)

for i in range(n_files):
    frames[i], header=fits.getdata(path[i],header=True)
    #remove cross talk for each file: do we need to do?
    frames[i]=xtalk(frames[i], aplpha=1.4e-5)

#J flats should be in expected order, but user should check this!!!
#average frames taken in the same conditions
Foff=frames[0]+frames[7]
Fon=frames[3]+frames[4]
Foff_mask=frames[1]+frames[6]
Fon_mask=frames[2]+frames[5]

#create the bias correction for the flat-on according to the Lidman technique
#check if it is aixs=1 or 0!!!
shade_off_mask=np.median(Foff_mask[:,50:150],axis=1)
shade_on_mask=np.median(Fon_mask[:,50:150],axis=1)

#compute difference in shade between masked and non-masked flats
ds_off=np.median((Foff_mask-Foff)[:,500:600],axis=1)
ds_on=np.median((Fon_mask-Fon)[:,500:600],axis=1)

#compute real shade patterns
shade_off = shade_off_mask - ds_off
shade_on = shade_on_mask - ds_on

#remove shade from flats
for i in range(1024):
   Foff[:,i]-=shade_off
   Fon[:,i]-=shade_on

#subtract Foff from Fon
mflat=Fon-Foff
fits.writeto(out_dir+'/mflat' + '.fits', mflat, header, output_verify='ignore', overwrite=True)
# fits.writeto(out_dir+'/mflat_21' + band + '.fits', mflat, header, output_verify='ignore')

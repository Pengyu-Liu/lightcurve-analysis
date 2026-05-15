#preprocessing: sky subtraction
import numpy as np
import glob
from astropy.io import fits
from tqdm import tqdm
from lightcurve_reduction.special_functions import xtalk
import os, errno

in_dir='/Users/data/SOFI/2023_05_10/2M1119-1137Ks/raw'
out_dir='/Users/data/SOFI/2023_05_10/2M1119-1137Ks/xtalk'

try:
    os.mkdir(out_dir)
    print('Directory ' + out_dir + ' created successfully')
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Warning: Directory ' + out_dir + ' already exists')
    else:
        raise

path=sorted(glob.glob(in_dir+'/*.fits'))
n_files=len(path)
for i in tqdm(range(n_files)):
    frame, header=fits.getdata(path[i],header='True')
    frame=xtalk(frame, aplpha=1.4e-5)
    new=path[i].replace(in_dir, out_dir).replace('.fits', '_xtalk.fits')
    fits.writeto(new, frame, header,output_verify='ignore',overwrite=True)

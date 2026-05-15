#check the location fit of all stars
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from photutils.aperture import CircularAperture
import os, errno
import shutil
from astropy.io import fits
from astropy.visualization import SqrtStretch, ImageNormalize

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (7, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

target_name = '2M0619-2903JsHKs'
date = '2022_02_19'
in_dir= '/Users/data/SOFI/' + date + '/' + target_name+'/reduced_Js'
image_dir = '/Users/data/SOFI/' + date + '/'+ target_name + '/skysub_Js'

plot_dir = in_dir + '/position_plots'
try:
    os.mkdir(plot_dir)
    print('Directory' + plot_dir + ' created successfully')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


path=sorted(glob.glob(image_dir+'/*skysub.fits'))
n_files=len(path)
#read in fitted locations from phot.py
#axis=0, files; axis=1, stars
xy_found=np.load(in_dir+'/xy_found.npy')
nt, ns, _ = xy_found.shape
norm=ImageNormalize(stretch=SqrtStretch(), vmin=-200, vmax=200)

for i in range(nt):
    plt.figure()
    img = fits.getdata(path[i], header=False)
    positions=np.transpose((xy_found[i, :, 0], xy_found[i, :,1]))
    apertures=CircularAperture(positions, r=8.)
    plt.imshow(img,norm=norm)
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    for j in range(ns):
        plt.text(xy_found[i, j, 0]+5, xy_found[i, j, 1]+5, '{:}'.format(j), fontsize=10)
    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    plt.title('{:3d}'.format(i))
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(plot_dir + '/{:03d}.png'.format(i), dpi=150)
    plt.close()

fp_in = plot_dir +'/*.png'
fp_out = in_dir+'/position_check'+'.gif'
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

shutil.rmtree(plot_dir)

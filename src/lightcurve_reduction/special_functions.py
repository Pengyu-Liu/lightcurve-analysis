import numpy as np
from photutils.aperture import CircularAperture,aperture_photometry,CircularAnnulus
from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch, ImageNormalize
import imageio
import os, errno
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import copy
import shutil

mpl.rc('image', interpolation='nearest', origin='lower')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (7, 7),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize': 15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

__epsilon = np.finfo(float).eps
__delta = 5.0e-7

def robust_sigma(inputData, Zero=False, axis=None, dtype=None, keepdims=False, return_mask=False):
    """Robust Sigma

    Based on the robust_sigma function from the AstroIDL User's Library.

    Calculate a resistant estimate of the dispersion of a distribution.

    Use the median absolute deviation as the initial estimate, then weight
    points using Tukey's Biweight. See, for example, "Understanding Robust
    and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    Wiley & Sons, 1983, or equation 9 in Beers et al. (1990, AJ, 100, 32).

    Parameters
    ==========
    inputData : ndarray
        The input data.

    Keyword Args
    ============
    axis : None or int or tuple of ints, optional
        Axis or axes along which the deviation is computed. The
        default is to compute the deviation of the flattened array.

        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
        This is the equivalent of reshaping the input data and then taking
        the standard devation.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    return_mask : bool
        If set to True, then only return boolean array of good (1) and
        rejected (0) values.

	"""

    inputData = np.array(inputData)

    if np.isnan(inputData).sum() > 0:
        medfunc = np.nanmedian
        meanfunc = np.nanmean
    else:
        medfunc = np.median
        meanfunc = np.mean

    if axis is None:
        data = inputData.ravel()
    else:
        data = inputData

    if type(data).__name__ == "MaskedArray":
        data = data.compressed()
    if dtype is not None:
        data = data.astype(dtype)

    # Scale factor to return result equivalent to standard deviation.
    sig_scale = 0.6744897501960817

    # Calculate the median absolute deviation
    if Zero:
        data0 = 0.0
    else:
        data0 = medfunc(data, axis=axis, keepdims=True)
    absdiff = np.abs(data-data0)
    medAbsDev = medfunc(absdiff, axis=axis, keepdims=True) / sig_scale
    mask = medAbsDev < __epsilon
    if np.any(mask):
        medAbsDev[mask] = (meanfunc(absdiff, axis=axis, keepdims=True))[mask] / 0.8

    # These will be set to 0 later
    mask0 = medAbsDev < __epsilon

    u = (data-data0) / (6.0 * medAbsDev)
    u2 = u**2.0
    good = u2 <= 1.0

    if return_mask:
        return good & ~np.isnan(data)

    # These values will be set to NaN later
    # if fewer than 3 good points to calculate stdev
    ngood = good.sum(axis=axis, keepdims=True)
    mask_nan = ngood < 2
    if mask_nan.sum() > 0:
        print("WARNING: NaN's will be present due to weird distributions")

    # Set bad points to NaNs
    u2[~good] = np.nan

    numerator = np.nansum( (data - data0)**2 * (1.0 - u2)**4.0, axis=axis, keepdims=True)
    nElements = len(data) if axis is None else data.shape[axis]
    denominator = np.nansum( (1.0 - u2) * (1.0 - 5.0*u2), axis=axis, keepdims=True)
    sigma = np.sqrt( nElements*numerator / (denominator*(denominator-1.0)) )

    sigma[mask0] = 0
    sigma[mask_nan] = np.nan

    if len(sigma)==1:
        return sigma[0]
    elif not keepdims:
        return np.squeeze(sigma)
    else:
        return sigma

def xtalk(im, aplpha=1.4e-5):
    #implement xtalk correction for SoFI data
    #input image is replaced with x_talk corrected image
    #sum along rows
    ny,nx=im.shape
    row_sumA=np.ones_like(im)
    for i in range(ny):
        row_sumA[i,:]*=np.sum(im[i,:])
    row_sumB=np.copy(row_sumA)
    row_sumB[512:, :]=np.copy(row_sumA[:512, :])
    row_sumB[:512, :] = np.copy(row_sumA[512:, :])
    im_new=im-aplpha*(row_sumA+row_sumB)
    return im_new

def split_sets(vect,gap):
    #splits a vector with gaps larger than "gap" into parts
    #returns:
    #parts:integer vector of length vsplit, indicating which part element belongs to
    #nparts: number of parts vsplit is split into
    #starts: array containing index locations of the first element of each set
    #lens-length of each vector in set
    num=vect.size
    parts=np.zeros(num, dtype='int')
    starts=[0]
    cnt=0
    for i in range(1,num):
        if abs(vect[i]-vect[i-1]) > gap:
            cnt+=1
            starts.append(i)
        parts[i]=cnt
    nparts=cnt+1
    lens=np.zeros(nparts, dtype='int')
    for i in range(nparts):
        if i<nparts-1:
           lens[i]=starts[i+1]-stars[i]
        else:
           lens[i]=num-starts[i]
    return parts, nparts, starts, lens

def aper_sub_sky(data,positions,apr_rad,sky_rad,bad_pixel=None):
    #aperture photometry for stars in positions
    aperture = CircularAperture(positions, r=apr_rad)
    #calculate backgound level in an annulus around the star
    annulus_aperture = CircularAnnulus(positions, r_in=sky_rad[0], r_out=sky_rad[1])
    annulus_masks = annulus_aperture.to_mask(method='center')
    bkg_median = []
    #make up a bad pixel mask
    if bad_pixel is None:
        bad_pixel=np.zeros_like(data,dtype='bool')
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        #bad pixel mask
        annulus_badpixel_mask=mask.multiply(~bad_pixel).astype('bool')
        #no need >0?
        annulus_data_1d = annulus_data[(mask.data > 0)*annulus_badpixel_mask]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d,sigma=3.0)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    #do aperture photometry
    phot = aperture_photometry(data, aperture,mask=bad_pixel)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    return phot

def surf2d(x,y,p):
    z=p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y\
      + p[6]*x**3 + p[7]*y**3 +p[8]*x**2*y + p[9]*y**2*x
    return z

def surf_polyfit2d(x,y,z):
    A = np.array([x*0+1, x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, y**2*x]).T
    #coeff, r, rank, s
    return np.linalg.lstsq(A, z)


def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    #coped from https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
        s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)



def check_position(files, out_dir, xy_found):
    plot_dir = out_dir + '/position_plots'
    try:
        os.mkdir(plot_dir)
        print('Directory' + plot_dir + ' created successfully')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    nt, ns, _ = xy_found.shape
    norm=ImageNormalize(stretch=SqrtStretch(), vmin=-200, vmax=200)

    for i in range(nt):
        plt.figure()
        img = files[i]
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
    fp_out = out_dir+'/position_check'+'.gif'
    img=[]
    for f in sorted(glob.glob(fp_in)):
        temp = Image.open(f)
        img.append(copy.deepcopy(temp))
        temp.close()
    imageio.mimsave(fp_out, img, duration = 0.04)

    shutil.rmtree(plot_dir)

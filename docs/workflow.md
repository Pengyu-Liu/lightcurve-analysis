# Procedures to Reduce the SoFI Data

This file records the reduction sequence used by the scripts in this
repository. The parameters in each script should be checked and edited for
each dataset before running the next stage.

## 0. Uncompress Files

Uncompress the raw files, for example with `extract.sh`.

## 1. Remove Cross Talk

Script: `preprocess/remove_xtalk.py`

Input: raw science images, `in_dir = /raw`

Output: cross-talk-corrected science images, `out_dir = /xtalk`

## 2. Reduce Special Dome Flat

Script: `preprocess/flat_field.py`

Input: 8 special dome flats, `in_dir = /flat`

Output: reduced flat, usually `/mflat.fits`

## 3. Reduce Illumination Correction

Script: `preprocess/illum_cor.py`

Input: 16 illumination-correction files, `in_dir = /std`

Output: reduced illumination correction, usually `/illum.fits`

## 4. Subtract Sky by Nods

Script: `photometry/skysub_nod.py`

Subtract sky by nods with the flat field and illumination correction for the
science frames.

Input: `/xtalk`, `/mflat.fits`, `/illum.fits`, and optionally
`/bad_pixel_map.fits` from ESO

Output: `/skysub`

## 5. Detect Sources and Run Aperture Photometry

Script: `photometry/phot.py`

Input: `/skysub`

Output: photometry table and helper files, `out_dir = /reduced`

## 6. Select Reference Stars and Detrend Light Curves

Script: `photometry/light_curve.py`

Input: `/reduced`

Output: light curves, `out_dir = /reduced`

## 7. Calculate Periodogram

Script: `periodogram/period.py`

Input: detrended light curves, `in_dir = /reduced`

Output: periodogram, `out_dir = /reduced/plots`

## 8. Calculate Period Sensitivity with Signal Injection

Script: `sensitivity/sensitivity.py`

Input: light curves, `in_dir = /reduced`

Output: sensitivity map, `out_dir = /reduced`

Plot the map with `sensitivity/sensitivity_plot.py`.

## 9. Analyse Light-Curve Correlation

Script: `periodogram/correlation.py`

Input: observation conditions and detrended light curves, `in_dir = /reduced`

Output: correlation plot, `out_dir = /reduced/plots`

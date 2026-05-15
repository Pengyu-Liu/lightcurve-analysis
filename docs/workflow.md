# Workflow Notes

This repository follows the original reduction sequence used by the scripts.
The parameters in each script should be checked for every dataset before
running the next stage.

1. Remove detector cross talk with `preprocess/remove_xtalk.py`.
2. Build the flat field with `preprocess/flat_field.py`.
3. Build the illumination correction with `preprocess/illum_cor.py`.
4. Subtract sky frames with `photometry/skysub_nod.py`.
5. Detect sources and run aperture photometry with `photometry/phot.py`.
6. Select reference stars and detrend light curves with `photometry/light_curve.py`.
7. Run periodogram analysis with `periodogram/period.py` or `periodogram/period_BGLS.py`.
8. Run injection and recovery sensitivity tests with `sensitivity/sensitivity.py`.
9. Analyse correlations with `periodogram/correlation.py`.

The repository is intended to document and share the analysis approach, not to
provide a fully automatic command-line pipeline.

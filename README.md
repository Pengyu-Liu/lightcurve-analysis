# Light-Curve Reduction Example Code

This repository contains research code for reducing near-infrared telescope
image sequences into differential light curves and time-series diagnostics for interested objects. The raw data are series of telescope images taken
continuously over several hours.


## Repository Layout

```text
preprocess/                 Calibration and preprocessing scripts
photometry/                 Sky subtraction, aperture photometry, light curves
periodogram/                Periodogram and correlation analysis
sensitivity/                Injection/recovery sensitivity measurements
src/lightcurve_reduction/   Shared Python helpers used by the scripts
examples/                   Example dataset-specific parameters
docs/                       Additional workflow notes
```

## Installation

Create and activate a Python environment, then install the repository in
editable mode:

```bash
python -m pip install -e .
```

The dependency list is recorded in both `pyproject.toml` and `requirements.txt`.

If you prefer installing dependencies manually first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset-Specific Parameters

Before running a script, inspect the variables near the top of the file. In
this project, many parameters naturally change from dataset to dataset,
including:

- input and output directories
- target name and observing date
- aperture radii
- selected aperture radius
- detector cross-talk coefficient
- source-selection thresholds
- reference-star choices
- binning and period-search settings

See `examples/example_paths.py` for a small template of common path and
analysis parameters. The full reduction sequence is described in
`docs/workflow.md`.

## Citation

If you use this repository in research, please cite:

Liu, P. et al. 2024, "A near-infrared variability survey of young
planetary-mass objects", Monthly Notices of the Royal Astronomical Society,
527, 6624-6674. doi: 10.1093/mnras/stad3502

## License

This repository currently uses the MIT License.

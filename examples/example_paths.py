"""Example dataset-specific parameters to copy into reduction scripts.

The scripts in this repository are intentionally research-code examples:
paths, target names, aperture choices, and detector-specific settings often
need to be edited for each dataset.
"""

TARGET_NAME = "J0200-5105"
DATE = "2021_10_20"

BASE_DIR = f"/Users/example/data/SOFI/{DATE}/{TARGET_NAME}"
RAW_DIR = f"{BASE_DIR}/raw"
XTALK_DIR = f"{BASE_DIR}/xtalk"
SKYSUB_DIR = f"{BASE_DIR}/skysub"
REDUCED_DIR = f"{BASE_DIR}/reduced"

APERTURE_RADII = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
SELECTED_APERTURE = 4.5

XTALK_ALPHA = 1.4e-5

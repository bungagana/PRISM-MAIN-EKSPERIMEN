"""Fixed constants for the PRISM pipeline.

d_ph(phi) and beta_FDA(phi) are taken directly from the paper's own
disclosed calibration results (Fig. 3 / Table 2) rather than re-derived by
grid search: they are physically-motivated constants (Beer-Lambert optical
depth deviation, FDA low-frequency bandwidth) that the paper explicitly
publishes as reproducible artifacts ("All calibrated parameters are
provided in Table 2", Section 4.7). Using them directly is the most
faithful and least noisy reproduction available without the original
authors' exact source images.

kappa_c (background attenuation) and lambda_c,phi (Yeo-Johnson shape
parameter) are NOT hardcoded here - they are estimated from data at
calibration time (Eq. 9-10 and Eq. 15's MLE step), exactly as Algorithm 1
specifies, because they are dataset-dependent statistics, not universal
physical constants.
"""
from pathlib import Path

IMAGE_SIZE = 224
PHASES = ["Lag", "Log", "Stationary", "Death"]

# Folder-name mapping: canonical phase name -> subfolder name on disk.
TARGET_PHASE_DIR = {"Lag": "Lag", "Log": "Log", "Stationary": "Stationer", "Death": "Death"}
SOURCE_PHASE_DIR = {"Lag": "lag", "Log": "log", "Stationary": "stationer", "Death": "death"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Paper's own quota for the target domain (Section 3.1: 2,123 images total).
MAX_TARGET_PER_PHASE = {"Lag": 600, "Log": 600, "Stationary": 600, "Death": 323}

# Paper Fig. 3 / Section 3.3.2: effective optical path length d_ph(phi).
D_PHASE = {"Lag": -0.05, "Log": -0.01, "Stationary": -0.02, "Death": 0.10}

# Paper Table 2: FDA low-frequency bandwidth beta_phi.
BETA_FDA = {"Lag": 0.07, "Log": 0.09, "Stationary": 0.10, "Death": 0.14}

# Eq. 12: spatial modulation coefficient beta_s.
BETA_SPATIAL = 0.15

SEED = 42
SPLIT_SEED = 165
TEST_FRACTION = 0.30
N_SYNTH_PER_PHASE = 500

# How many target_cal images to use for estimating PCCA target moments and
# the Yeo-Johnson lambda (larger = less noisy estimate of the calibration
# distribution's shape).
TARGET_CAL_MOMENT_CAP = 300

CLASSIFIER_LABEL_MAP = {p: i for i, p in enumerate(PHASES)}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"

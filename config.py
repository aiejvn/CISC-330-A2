"""
config.py - shared constants for Tasks 1 and 2. Lengths in mm, angles in degrees.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import math
import os
import numpy as np

# ---- Ground-truth tool model from the assignment, in F_mod (mm) ----------------
MARKER_NAMES = ["M1", "M2", "M3"]
MARKERS_MOD_mm = np.array([[2.0, 18.0, 0.0],      # M1_mod
                           [8.0, 18.0, 0.0],      # M2_mod
                           [2.0, 24.0, 0.0]])     # M3_mod
TIP_MOD_mm = np.array([0.0, 0.0, 0.0])            # P_mod
AXIS_MOD = np.array([0.0, -1.0, 0.0])             # V_mod, points from the tip away from the handle

# ---- Marker noise ------------------------------------------------------------------
# Zero-mean Gaussian noise with the same sigma on x, y and z. Then |error|^2 / sigma^2
# follows a chi-square distribution with 3 degrees of freedom, whose 95% point is 7.8147,
# so 95% of the 3-D errors are shorter than 1.0 mm when sigma = 1.0 / sqrt(7.8147).
MARKER_ERROR_RADIUS_95_mm = 1.0
CHI2_3DOF_95 = 7.814727903251178
MARKER_SIGMA_mm = MARKER_ERROR_RADIUS_95_mm / math.sqrt(CHI2_3DOF_95)   # 0.358 mm
RANDOM_SEED = 330

# ---- Simulated calibration motions (the tracker frame is F_mod) -------------------
START_ROTATION_X_deg = 20.0                       # starting pose: turn about x, then about z,
START_ROTATION_Z_deg = 30.0
PIVOT_POINT_TRACKER_mm = np.array([100.0, 50.0, 200.0])   # and move the tip here
N_POSES = 100                                     # poses per motion
CONE_HALF_ANGLE_deg = 45.0                        # pivot: largest tilt (a 90 deg cone)
TWIST_RANGE_deg = 30.0                            # pivot: twist about the shaft, within +-30 deg
SPIN_RANGE_deg = 360.0                            # spin: one full turn

# ---- Repeated noisy calibrations ------------------------------------------------------
N_REPEATS = 1000
N_REPEATS_SWEEP = 200                             # per row of the motion-setting tables
CONE_SWEEP_deg = [10, 20, 30, 45, 60]             # cone half-angles
SPIN_SWEEP_deg = [30, 60, 90, 180, 360]           # spin ranges
POSES_SWEEP = [25, 50, 100, 200, 400]             # numbers of poses

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

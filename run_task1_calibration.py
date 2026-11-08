"""
Task 1: tool calibration with noisy marker observations.

Runs the calibration 1000 times at our settings, prints the tip and axis errors and the
numbers that explain them, then changes one motion setting at a time (cone half-angle,
spin range, number of poses). Saves outputs/task1_calibration_results.txt and the two
figures. Takes about 50 s. The exact-data checks are in tests/test_task1_calibration.py.

Run from the A2 folder:   python3 run_task1_calibration.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import numpy as np
import config
import simulate
import figures
from calibration import tool_in_marker_frame
from noise import add_marker_noise, error_statistics
from noisy_calibration import run_noisy_calibrations, frame_rotation_errors_deg

output_lines = []


def say(text=""):
    """Print a line and keep it for the output file."""
    print(text)
    output_lines.append(text)


def vector_text(vector, decimals=3):
    return "(" + ", ".join(f"{value + 0.0:.{decimals}f}" for value in np.round(vector, decimals)) + ")"


os.makedirs(config.OUTPUT_DIR, exist_ok=True)
tip_true_marker_mm, axis_true_marker = tool_in_marker_frame(config.MARKERS_MOD_mm, config.TIP_MOD_mm, config.AXIS_MOD)

say("Task 1: tool calibration with noisy marker observations")
say(f"  true tool in the marker frame: p_m = {vector_text(tip_true_marker_mm, 0)} mm, v_m = {vector_text(axis_true_marker, 0)}")
say(f"  marker noise: sigma = {config.MARKER_SIGMA_mm:.3f} mm on each of x, y, z "
    f"(95% of 3-D errors < {config.MARKER_ERROR_RADIUS_95_mm:.1f} mm)")
say(f"  pivot: {config.N_POSES} poses, cone half-angle {config.CONE_HALF_ANGLE_deg:.0f} deg, "
    f"twist +-{config.TWIST_RANGE_deg:.0f} deg;  spin: {config.N_POSES} poses over {config.SPIN_RANGE_deg:.0f} deg")
say(f"  {config.N_REPEATS} repeats; the motions stay the same and only the noise changes (seed {config.RANDOM_SEED})")

# ---- 1. Main run ----------------------------------------------------------------------
results = run_noisy_calibrations(config.CONE_HALF_ANGLE_deg, config.SPIN_RANGE_deg, config.N_POSES, config.N_REPEATS)
tip_statistics = error_statistics(results["tip_error_mm"])
axis_statistics = error_statistics(results["axis_error_deg"])
say()
say(f"1. Errors over {config.N_REPEATS} repeats            mean      p95")
say(f"   tool-tip error (mm)          {tip_statistics['mean']:9.3f}{tip_statistics['p95']:9.3f}")
say(f"   tool-axis error (deg)        {axis_statistics['mean']:9.3f}{axis_statistics['p95']:9.3f}")

# Why the tip error is about 1 mm: the noisy marker frames pull the tip toward the markers.
mean_error_vector_mm = results["tip_error_vector_mm"].mean(axis=0)
tip_estimates_mm = tip_true_marker_mm + results["tip_error_vector_mm"]
say()
say(f"   mean tip error vector {vector_text(mean_error_vector_mm)} mm, length {np.linalg.norm(mean_error_vector_mm):.3f} mm")
say(f"   distance of the tip from the marker frame origin: true {np.linalg.norm(tip_true_marker_mm):.2f} mm, "
    f"computed {np.linalg.norm(tip_estimates_mm, axis=1).mean():.2f} mm on average")

# How far the noise turns each pose's marker frame (one noisy draw of the pivot markers).
rng = np.random.default_rng(config.RANDOM_SEED)
F_pivot, markers_pivot_mm, F_spin, markers_spin_mm = simulate.calibration_motions(
    config.N_POSES, config.CONE_HALF_ANGLE_deg, config.SPIN_RANGE_deg, rng)
noisy_pivot_mm = add_marker_noise(markers_pivot_mm, config.MARKER_SIGMA_mm, rng)
say(f"   marker frame turned by the noise: {frame_rotation_errors_deg(markers_pivot_mm, noisy_pivot_mm).mean():.1f} deg "
    "on average")

say("   plane-normal error, median:  " + ",  ".join(
    f"{name} {error_deg:.2f} deg" for name, error_deg in zip(config.MARKER_NAMES, np.median(results["normal_error_deg"], axis=0))))

# ---- 2. One motion setting changed at a time -----------------------------------------------
n_sweep = config.N_REPEATS_SWEEP
say()
say(f"2. Tool-tip error vs cone half-angle ({n_sweep} repeats per row)")
say("   half-angle (deg)    mean (mm)   p95 (mm)")
cone_rows = []
for cone_deg in config.CONE_SWEEP_deg:
    statistics = error_statistics(
        run_noisy_calibrations(cone_deg, config.SPIN_RANGE_deg, config.N_POSES, n_sweep)["tip_error_mm"])
    cone_rows.append((cone_deg, statistics))
    say(f"   {cone_deg:<16d}{statistics['mean']:12.3f}{statistics['p95']:11.3f}")

say()
say(f"3. Tool-axis error vs spin range ({n_sweep} repeats per row)")
say("   spin range (deg)    mean (deg)  p95 (deg)")
spin_rows = []
for spin_deg in config.SPIN_SWEEP_deg:
    statistics = error_statistics(
        run_noisy_calibrations(config.CONE_HALF_ANGLE_deg, spin_deg, config.N_POSES, n_sweep)["axis_error_deg"])
    spin_rows.append((spin_deg, statistics))
    say(f"   {spin_deg:<16d}{statistics['mean']:12.3f}{statistics['p95']:11.3f}")

say()
say(f"4. Tool-tip and tool-axis error vs number of poses ({n_sweep} repeats per row)")
say("   poses     tip mean (mm)  tip p95 (mm)   axis mean (deg)  axis p95 (deg)")
poses_rows = []
for n_poses in config.POSES_SWEEP:
    sweep = run_noisy_calibrations(config.CONE_HALF_ANGLE_deg, config.SPIN_RANGE_deg, n_poses, n_sweep)
    tip_row, axis_row = error_statistics(sweep["tip_error_mm"]), error_statistics(sweep["axis_error_deg"])
    poses_rows.append((n_poses, tip_row, axis_row))
    say(f"   {n_poses:<10d}{tip_row['mean']:13.3f}{tip_row['p95']:14.3f}{axis_row['mean']:17.3f}{axis_row['p95']:16.3f}")

# ---- 3. Figures ----------------------------------------------------------------------------
figures.plot_motions(F_pivot, markers_pivot_mm, F_spin, markers_spin_mm,
                     os.path.join(config.OUTPUT_DIR, "task1_motions.png"))
figures.plot_motion_sweeps(cone_rows, spin_rows, poses_rows,
                           os.path.join(config.OUTPUT_DIR, "task1_motion_sweeps.png"))

with open(os.path.join(config.OUTPUT_DIR, "task1_calibration_results.txt"), "w") as f:
    f.write("\n".join(output_lines) + "\n")
print("Saved outputs/task1_calibration_results.txt, task1_motions.png, task1_motion_sweeps.png")

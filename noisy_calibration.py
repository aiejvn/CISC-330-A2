"""
noisy_calibration.py - Task 1 with noisy markers: repeat the calibration many times and
collect the tip and axis errors.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
import config
import simulate
from calibration import calibrate_tip_pivot, calibrate_axis_spin, tool_in_marker_frame
from frames import marker_frames
from geometry import transform_direction, angle_between_deg, rotation_angle_deg
from noise import add_marker_noise


def run_noisy_calibrations(cone_half_angle_deg, spin_range_deg, n_poses, n_repeats):
    """
    Repeat the tip and axis calibration n_repeats times. The motions stay the same and
    only the marker noise changes. Errors are measured against the true tool in the
    marker frame.
    Returns a dict of arrays, one entry per repeat:
      tip_error_mm         |p_m computed - p_m true|
      tip_error_vector_mm  p_m computed - p_m true, (n_repeats, 3)
      axis_error_deg       angle between v_m computed and v_m true
      normal_error_deg     angle between each plane normal n_A, n_B, n_C and the true axis, (n_repeats, 3)
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    _, markers_pivot_mm, _, markers_spin_mm = simulate.calibration_motions(
        n_poses, cone_half_angle_deg, spin_range_deg, rng)
    tip_true_marker_mm, axis_true_marker = tool_in_marker_frame(
        config.MARKERS_MOD_mm, config.TIP_MOD_mm, config.AXIS_MOD)
    axis_true_tracker = transform_direction(simulate.start_pose(), config.AXIS_MOD)

    results = {"tip_error_mm": np.zeros(n_repeats),
               "tip_error_vector_mm": np.zeros((n_repeats, 3)),
               "axis_error_deg": np.zeros(n_repeats),
               "normal_error_deg": np.zeros((n_repeats, 3))}
    for repeat in range(n_repeats):
        noisy_pivot_mm = add_marker_noise(markers_pivot_mm, config.MARKER_SIGMA_mm, rng)
        noisy_spin_mm = add_marker_noise(markers_spin_mm, config.MARKER_SIGMA_mm, rng)
        tip_marker_mm, _ = calibrate_tip_pivot(noisy_pivot_mm)
        axis_marker, _, normals_tracker = calibrate_axis_spin(noisy_spin_mm, tip_marker_mm)

        results["tip_error_vector_mm"][repeat] = tip_marker_mm - tip_true_marker_mm
        results["tip_error_mm"][repeat] = np.linalg.norm(tip_marker_mm - tip_true_marker_mm)
        results["axis_error_deg"][repeat] = angle_between_deg(axis_marker, axis_true_marker)
        for marker, normal in enumerate(normals_tracker):
            normal_same_side = normal if normal @ axis_true_tracker >= 0 else -normal   # the sign of a normal does not matter here
            results["normal_error_deg"][repeat, marker] = angle_between_deg(normal_same_side, axis_true_tracker)
    return results


def frame_rotation_errors_deg(markers_true_mm, markers_noisy_mm):
    """Angle (deg) by which the noise turns the marker frame, for every pose (n_poses,)."""
    R_true = marker_frames(markers_true_mm)[:, :3, :3]
    R_noisy = marker_frames(markers_noisy_mm)[:, :3, :3]
    return np.array([rotation_angle_deg(Rn @ Rt.T) for Rn, Rt in zip(R_noisy, R_true)])

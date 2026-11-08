"""
calibration.py - Task 1: calibrate the tool tip (pivot) and the tool axis (spin).

In every pose the tracker reports the three markers, which give the transformation
F_tracker_from_marker = (R_i, t_i). The unknowns are two constants of the tool in the
marker frame: the tip p_m and the axis v_m. After calibration a tracker reading gives the
tip p_h = R_i p_m + t_i and the axis v_h = R_i v_m.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize, invert_rigid, apply_transform, transform_direction
from frames import frame_from_three_points, marker_frames
from fitting import fit_plane


def pivot_equations(F_tracker_from_marker):
    """
    Pivot equations for all poses. The tip stays at the pivot point p_h, so in pose i
        R_i p_m + t_i = p_h,   i.e.   [R_i  -I] [p_m; p_h] = -t_i.
    Stacking the n poses gives A x = b with A (3n, 6), x = [p_m; p_h] and b (3n,).
    """
    n_poses = len(F_tracker_from_marker)
    A = np.zeros((3 * n_poses, 6))
    b = np.zeros(3 * n_poses)
    for i, F in enumerate(F_tracker_from_marker):
        rows = slice(3 * i, 3 * i + 3)
        A[rows, :3] = F[:3, :3]
        A[rows, 3:] = -np.eye(3)
        b[rows] = -F[:3, 3]
    return A, b


def calibrate_tip_pivot(markers_tracker_mm):
    """
    Tool-tip calibration from a pivot motion.
    markers_tracker_mm: (n_poses, 3, 3), the markers of every pose in the tracker frame.
    Returns (tip_marker_mm, pivot_tracker_mm): p_m in the marker frame and p_h in the tracker frame.
    A has 3n rows and 6 columns, so we solve A x = b with the pseudo-inverse, which gives
    the least-squares solution over all poses.
    """
    A, b = pivot_equations(marker_frames(markers_tracker_mm))
    x = np.linalg.pinv(A) @ b
    return x[:3], x[3:]


def calibrate_axis_spin(markers_tracker_mm, tip_marker_mm):
    """
    Tool-axis calibration from a spin motion.
    markers_tracker_mm: (n_poses, 3, 3), the markers of every pose in the tracker frame.
    tip_marker_mm: p_m from the tip calibration, used only to choose the sign of the axis.
    Returns (axis_marker, axis_tracker, normals_tracker): v_m, v_h and the plane normals
    n_A, n_B, n_C (one row per marker).
    """
    # Each marker moves on a circle perpendicular to the axis, so the normal of its plane
    # is the axis direction.
    normals_tracker = np.array([fit_plane(markers_tracker_mm[:, marker, :])[1] for marker in range(3)])

    # A plane normal has no preferred sign. Flip n_A and n_C to the side of n_B before
    # averaging: M2 moves on the largest circle, so n_B is the most reliable normal.
    for marker in (0, 2):
        if normals_tracker[marker] @ normals_tracker[1] < 0:
            normals_tracker[marker] = -normals_tracker[marker]
    axis_tracker = normalize(normals_tracker.mean(axis=0))

    # Bring v_h into the marker frame of every pose (v_mi = R_i^T v_h) and average.
    axis_marker_per_pose = np.array([transform_direction(invert_rigid(F), axis_tracker)
                                     for F in marker_frames(markers_tracker_mm)])
    axis_marker = normalize(axis_marker_per_pose.mean(axis=0))

    # The insertion direction points from the markers toward the tip: v_m . p_m > 0.
    if axis_marker @ tip_marker_mm < 0:
        axis_marker, axis_tracker, normals_tracker = -axis_marker, -axis_tracker, -normals_tracker
    return axis_marker, axis_tracker, normals_tracker


def tool_in_marker_frame(markers_mod_mm, tip_mod_mm, axis_mod):
    """Tip and axis of a tool given in F_mod, expressed in its marker frame: (p_m, v_m)."""
    F_marker_from_mod = invert_rigid(frame_from_three_points(*markers_mod_mm))
    return apply_transform(F_marker_from_mod, tip_mod_mm), transform_direction(F_marker_from_mod, axis_mod)


def tool_in_model_frame(markers_mod_mm, tip_marker_mm, axis_marker):
    """
    The inverse of tool_in_marker_frame: a calibration result (p_m, v_m) mapped into F_mod.
    These are Ptool and vtool, the inputs of the Task 2 registration.
    """
    F_mod_from_marker = frame_from_three_points(*markers_mod_mm)
    return apply_transform(F_mod_from_marker, tip_marker_mm), transform_direction(F_mod_from_marker, axis_marker)

"""
simulate.py - simulated tool motions for the Task 1 calibration.

The tracker frame coincides with F_mod, so the tool model is already in tracker
coordinates. A pose is a rigid motion of the tool, F_tracker_from_mod (4, 4), and the
markers the tracker reports in that pose are this motion applied to M1, M2, M3.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
import config
from geometry import translation, rotation_x, rotation_y, rotation_z, apply_transform


def start_pose():
    """
    F_tracker_from_mod of the starting pose: 20 deg about x, then 30 deg about z, then the
    tip moved to the pivot point. In the tool model the marker frame is parallel to F_mod,
    so this turn keeps the simulated marker frames from lining up with the tracker axes.
    """
    return (translation(config.PIVOT_POINT_TRACKER_mm)
            @ rotation_z(config.START_ROTATION_Z_deg)
            @ rotation_x(config.START_ROTATION_X_deg))


def pivot_poses(n_poses, cone_half_angle_deg, twist_range_deg, rng):
    """
    Pivot about the tool tip, which sits in a dimple. Each pose turns the tool about its
    tip (the F_mod origin): the shaft is tilted by theta toward a direction phi, and the
    tool is twisted by psi about its own shaft. The tip stays at the pivot point, and each
    marker moves on a sphere around it.

    phi is uniform in [0, 360) deg so the tilts cover every direction of the cone, and
    cos(theta) is uniform in [cos(cone_half_angle), 1] so the shaft directions are spread
    evenly over the cone. rotation_y(phi) rotation_x(theta) rotation_y(-phi) is a tilt by
    theta about a horizontal axis turned by phi.
    Returns F_tracker_from_mod for every pose, (n_poses, 4, 4).
    """
    tilt_direction_deg = rng.uniform(0.0, 360.0, n_poses)
    cos_tilt = rng.uniform(np.cos(np.radians(cone_half_angle_deg)), 1.0, n_poses)
    twist_deg = rng.uniform(-twist_range_deg, twist_range_deg, n_poses)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
    F_start = start_pose()
    return np.array([F_start @ rotation_y(phi) @ rotation_x(theta) @ rotation_y(-phi) @ rotation_y(psi)
                     for phi, theta, psi in zip(tilt_direction_deg, tilt_deg, twist_deg)])


def swing_poses(n_poses, swing_half_angle_deg):
    """Pivot about the tip, but tilting in one plane only (all turns about x_mod). Used in the rank test."""
    F_start = start_pose()
    return np.array([F_start @ rotation_x(theta)
                     for theta in np.linspace(-swing_half_angle_deg, swing_half_angle_deg, n_poses)])


def spin_poses(n_poses, spin_range_deg):
    """
    Spin about the tool axis, as inside a guide sleeve with a stop. Each pose turns the
    tool about its own axis line (through the tip, along y_mod), so each marker moves on
    a circle in a plane perpendicular to the axis. A full turn is sampled without
    repeating 0 = 360 deg.
    Returns F_tracker_from_mod for every pose, (n_poses, 4, 4).
    """
    if spin_range_deg >= 360.0:
        spin_deg = np.linspace(0.0, 360.0, n_poses, endpoint=False)
    else:
        spin_deg = np.linspace(0.0, spin_range_deg, n_poses)
    F_start = start_pose()
    return np.array([F_start @ rotation_y(psi) for psi in spin_deg])


def markers_in_poses(F_tracker_from_mod, markers_mod_mm):
    """Exact marker positions in every pose: (n_poses, 4, 4) and (3, 3) -> (n_poses, 3, 3)."""
    return np.array([apply_transform(F, markers_mod_mm) for F in F_tracker_from_mod])


def calibration_motions(n_poses, cone_half_angle_deg, spin_range_deg, rng):
    """
    The pivot and spin motions with their exact marker positions:
    (F_pivot, markers_pivot_mm, F_spin, markers_spin_mm), all in the tracker frame.
    """
    F_pivot = pivot_poses(n_poses, cone_half_angle_deg, config.TWIST_RANGE_deg, rng)
    F_spin = spin_poses(n_poses, spin_range_deg)
    return (F_pivot, markers_in_poses(F_pivot, config.MARKERS_MOD_mm),
            F_spin, markers_in_poses(F_spin, config.MARKERS_MOD_mm))

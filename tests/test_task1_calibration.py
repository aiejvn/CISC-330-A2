"""
Task 1 test: tool calibration with exact marker positions.

  1. The pivot motion keeps the tip fixed, and every marker stays on a sphere.
  2. The spin motion keeps the axis fixed, and every marker stays on a circle.
  3. With our motions the calibration recovers the ground-truth tool.
  4. A few poses are enough with exact data (5 pivot poses, 3 spin poses).
  5. Only tilting in different directions gives the pivot equations rank 6.
  6. A short spin arc still gives the exact axis with exact data.
True tool in the marker frame: p_m = (-40, -200, 0) mm, v_m = (0, -1, 0).

Run from the A2 folder:   python3 tests/test_task1_calibration.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import simulate
from checks import CheckList
from calibration import pivot_equations, calibrate_tip_pivot, calibrate_axis_spin, tool_in_model_frame
from frames import marker_frames
from geometry import apply_transform, transform_direction, angle_between_deg

TOL_mm, TOL_deg = 1e-9, 1e-9
TIP_TRUE_MARKER_mm = np.array([-40.0, -200.0, 0.0])     # P_mod (0,0,0) minus the marker frame origin (40,200,0)
AXIS_TRUE_MARKER = np.array([0.0, -1.0, 0.0])           # the marker frame is parallel to F_mod
PIVOT_POINT_mm = [100.0, 50.0, 200.0]
# The starting pose turns (0,-1,0) by 20 deg about x and then 30 deg about z:
# (sin30 cos20, -cos30 cos20, -sin20) = (0.469846, -0.813798, -0.342020).
AXIS_TRUE_TRACKER = np.array([0.469846, -0.813798, -0.342020])


def pivot_rank(markers_mm):
    """Rank of A in the pivot equations A x = b for the given marker positions (n, 3, 3)."""
    A, _ = pivot_equations(marker_frames(markers_mm))
    return np.linalg.matrix_rank(A)


checks = CheckList("Task 1 test: tool calibration with exact marker positions")
rng = np.random.default_rng(config.RANDOM_SEED)
F_pivot, markers_pivot_mm, F_spin, markers_spin_mm = simulate.calibration_motions(
    config.N_POSES, config.CONE_HALF_ANGLE_deg, config.SPIN_RANGE_deg, rng)

# ---- 1. Pivot motion ------------------------------------------------------------------------
checks.write()
checks.write(f"1. Pivot motion ({config.N_POSES} poses, cone half-angle {config.CONE_HALF_ANGLE_deg:.0f} deg)")
tips_mm = np.array([apply_transform(F, config.TIP_MOD_mm) for F in F_pivot])
checks.check("tip distance from (100,50,200), largest", 0.0, np.linalg.norm(tips_mm - PIVOT_POINT_mm, axis=1).max(),
             TOL_mm, "mm")
# Distance of each marker from the tip in the tool model: sqrt(20^2+180^2), sqrt(80^2+180^2), sqrt(20^2+240^2).
for marker, (name, radius_mm) in enumerate(zip(config.MARKER_NAMES, [181.108, 196.977, 240.832])):
    distances_mm = np.linalg.norm(markers_pivot_mm[:, marker, :] - PIVOT_POINT_mm, axis=1)
    checks.check(f"{name} distance from the tip, min and max", [radius_mm, radius_mm],
                 [distances_mm.min(), distances_mm.max()], 1e-3, "mm")
    checks.check(f"{name} distance, max - min over the poses", 0.0, distances_mm.max() - distances_mm.min(), TOL_mm, "mm")
shaft_start = transform_direction(simulate.start_pose(), [0, 1, 0])
tilts_deg = np.array([angle_between_deg(transform_direction(F, [0, 1, 0]), shaft_start) for F in F_pivot])
checks.check_true("largest tilt between 40 and 45 deg", 40.0 <= tilts_deg.max() <= 45.0, f"{tilts_deg.max():.2f} deg")

# ---- 2. Spin motion ---------------------------------------------------------------------------
checks.write()
checks.write(f"2. Spin motion ({config.N_POSES} poses over {config.SPIN_RANGE_deg:.0f} deg)")
tips_mm = np.array([apply_transform(F, config.TIP_MOD_mm) for F in F_spin])
axes_tracker = np.array([transform_direction(F, config.AXIS_MOD) for F in F_spin])
checks.check("tip distance from (100,50,200), largest", 0.0, np.linalg.norm(tips_mm - PIVOT_POINT_mm, axis=1).max(),
             TOL_mm, "mm")
checks.check("tool axis in the tracker frame", AXIS_TRUE_TRACKER, axes_tracker[0], 1e-5)
checks.check("tool axis, largest change over the poses", 0.0, np.abs(axes_tracker - axes_tracker[0]).max(), TOL_mm)
# Position along the axis and distance from it, from the tool model: M1 -180 and 20, M2 -180 and 80, M3 -240 and 20 mm.
for marker, (name, along_mm, radius_mm) in enumerate(zip(config.MARKER_NAMES, [-180, -180, -240], [20, 80, 20])):
    offsets_mm = markers_spin_mm[:, marker, :] - tips_mm
    along_axis_mm = offsets_mm @ axes_tracker[0]
    from_axis_mm = np.sqrt(np.sum(offsets_mm ** 2, axis=1) - along_axis_mm ** 2)
    checks.check(f"{name} position along the axis, min and max", [along_mm, along_mm],
                 [along_axis_mm.min(), along_axis_mm.max()], TOL_mm, "mm")
    checks.check(f"{name} distance from the axis, min and max", [radius_mm, radius_mm],
                 [from_axis_mm.min(), from_axis_mm.max()], TOL_mm, "mm")

# ---- 3. Exact data, our motions -----------------------------------------------------------------
checks.write()
checks.write("3. Exact data: the calibration recovers the ground-truth tool")
tip_marker_mm, pivot_tracker_mm = calibrate_tip_pivot(markers_pivot_mm)
axis_marker, axis_tracker, _ = calibrate_axis_spin(markers_spin_mm, tip_marker_mm)
checks.check("p_m, tool tip in the marker frame", TIP_TRUE_MARKER_mm, tip_marker_mm, TOL_mm, "mm")
checks.check("p_h, pivot point in the tracker frame", PIVOT_POINT_mm, pivot_tracker_mm, TOL_mm, "mm")
checks.check("v_m, angle to (0,-1,0)", 0.0, angle_between_deg(axis_marker, AXIS_TRUE_MARKER), TOL_deg, "deg")
checks.check("v_h, angle to the axis in the tracker frame", 0.0, angle_between_deg(axis_tracker, AXIS_TRUE_TRACKER),
             1e-3, "deg")
P_tool_mm, v_tool = tool_in_model_frame(config.MARKERS_MOD_mm, tip_marker_mm, axis_marker)
checks.check("tool tip mapped to F_mod", [0, 0, 0], P_tool_mm, TOL_mm, "mm")
checks.check("tool axis mapped to F_mod, angle to (0,-1,0)", 0.0, angle_between_deg(v_tool, config.AXIS_MOD), TOL_deg, "deg")

# ---- 4. Exact data, few poses ---------------------------------------------------------------------
checks.write()
checks.write("4. Exact data, first 5 pivot poses and 3 spin poses (0, 120, 240 deg)")
tip_small_mm, pivot_small_mm = calibrate_tip_pivot(markers_pivot_mm[:5])
markers_spin3_mm = simulate.markers_in_poses(simulate.spin_poses(3, 360.0), config.MARKERS_MOD_mm)
axis_small, _, _ = calibrate_axis_spin(markers_spin3_mm, tip_small_mm)
checks.check("p_m from 5 pivot poses", TIP_TRUE_MARKER_mm, tip_small_mm, TOL_mm, "mm")
checks.check("p_h from 5 pivot poses", PIVOT_POINT_mm, pivot_small_mm, TOL_mm, "mm")
checks.check("v_m from 3 spin poses, angle to (0,-1,0)", 0.0, angle_between_deg(axis_small, AXIS_TRUE_MARKER), TOL_deg, "deg")

# ---- 5. Rank of the pivot equations ---------------------------------------------------------------
checks.write()
checks.write("5. Rank of A (3n x 6). Rotations about one common axis leave the tip's position along that axis open.")
checks.check("cone, 100 poses", 6, pivot_rank(markers_pivot_mm), 0)
markers_swing_mm = simulate.markers_in_poses(simulate.swing_poses(50, 45.0), config.MARKERS_MOD_mm)
checks.check("swing in one plane, 50 poses", 5, pivot_rank(markers_swing_mm), 0)
checks.check("spin about the tool axis, 100 poses", 5, pivot_rank(markers_spin_mm), 0)
checks.check("cone, only 2 poses", 5, pivot_rank(markers_pivot_mm[:2]), 0)

# ---- 6. Short spin arc ------------------------------------------------------------------------------
checks.write()
checks.write("6. Exact data, 100 spin poses over a 30 deg arc")
markers_arc_mm = simulate.markers_in_poses(simulate.spin_poses(100, 30.0), config.MARKERS_MOD_mm)
axis_arc, _, _ = calibrate_axis_spin(markers_arc_mm, tip_marker_mm)
checks.check("v_m from a 30 deg arc, angle to (0,-1,0)", 0.0, angle_between_deg(axis_arc, AXIS_TRUE_MARKER), TOL_deg, "deg")

checks.finish("test_task1_calibration")

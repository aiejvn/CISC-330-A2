"""
Tests of the reusable utilities in geometry.py, frames.py and fitting.py, on small inputs
whose results can be worked out by hand.

Run from the A2 folder:   python3 tests/test_utilities.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checks import CheckList
from geometry import (translation, rotation_x, rotation_z, invert_rigid, transform_direction,
                      angle_between_deg, rotation_angle_deg)
from frames import frame_from_three_points
from fitting import fit_plane

TOL = 1e-12
checks = CheckList("Utilities test: geometry.py, frames.py, fitting.py")

# ---- frame_from_three_points ------------------------------------------------------------
checks.write()
checks.write("frame_from_three_points: i = normalize(B - A), k = normalize(i x (C - A)), j = k x i, O = (A + B + C)/3")
F = frame_from_three_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
checks.check("A(0,0,0), B(1,0,0), C(0,1,0): axes i, j, k", np.eye(3), F[:3, :3], TOL)
checks.check("A(0,0,0), B(1,0,0), C(0,1,0): origin O", [1 / 3, 1 / 3, 0], F[:3, 3], TOL, "mm")
# Tool model markers: B - A = (6,0,0) gives i = x, C - A = (0,6,0) gives k = z, so j = y.
F = frame_from_three_points([2, 18, 0], [8, 18, 0], [2, 24, 0])
checks.check("tool model markers: axes i, j, k", np.eye(3), F[:3, :3], TOL, "", "parallel to F_mod")
checks.check("tool model markers: origin O", [4, 20, 0], F[:3, 3], TOL, "mm")
# Markers turned 90 deg about z: i = (0,1,0), i x (-6,0,0) = (0,0,6) gives k = z, j = k x i = (-1,0,0).
F = frame_from_three_points([-18, 2, 0], [-18, 8, 0], [-24, 2, 0])
checks.check("turned markers: i", [0, 1, 0], F[:3, 0], TOL)
checks.check("turned markers: j", [-1, 0, 0], F[:3, 1], TOL)
checks.check("turned markers: k", [0, 0, 1], F[:3, 2], TOL)
checks.check("turned markers: origin O", [-20, 4, 0], F[:3, 3], TOL, "mm")

# ---- invert_rigid -------------------------------------------------------------------------
checks.write()
checks.write("invert_rigid: F^-1 F = I")
F_turned = np.array([[0, 0, 1, -40], [1, 0, 0, 60], [0, 1, 0, 100], [0, 0, 0, 1]], dtype=float)
checks.check("invert_rigid(F) @ F, 90 deg turns", np.eye(4), invert_rigid(F_turned) @ F_turned, TOL)
F_general = translation([5, -6, 7]) @ rotation_z(30) @ rotation_x(20)
checks.check("invert_rigid(F) @ F, general rotation", np.eye(4), invert_rigid(F_general) @ F_general, TOL)

# ---- transform_direction --------------------------------------------------------------------
checks.write()
checks.write("transform_direction: only the rotation acts on a direction")
checks.check("translation by (5,6,7) of (0,-1,0)", [0, -1, 0], transform_direction(translation([5, 6, 7]), [0, -1, 0]), TOL)
checks.check("rotation_z(90) of (1,0,0)", [0, 1, 0], transform_direction(rotation_z(90), [1, 0, 0]), TOL)
checks.check("rotation_z(90) of rows (1,0,0), (0,1,0)", [[0, 1, 0], [-1, 0, 0]],
             transform_direction(rotation_z(90), [[1, 0, 0], [0, 1, 0]]), TOL)

# ---- angle_between_deg and rotation_angle_deg -------------------------------------------------
checks.write()
checks.write("angle_between_deg and rotation_angle_deg")
checks.check("angle between x and y", 90.0, angle_between_deg([1, 0, 0], [0, 1, 0]), TOL, "deg")
checks.check("angle between opposite vectors", 180.0, angle_between_deg([0, -1, 0], [0, 2, 0]), TOL, "deg")
# (1,0,0) and (1,1e-7,0) are atan(1e-7) = 1e-7 rad = 5.729578e-6 deg apart.
checks.check("vectors 1e-7 rad apart", 5.729578e-6, angle_between_deg([1, 0, 0], [1, 1e-7, 0]), 1e-12, "deg")
checks.check("rotation angle of rotation_z(30)", 30.0, rotation_angle_deg(rotation_z(30)), 1e-9, "deg")
checks.check("rotation angle of rotation_x(20)", 20.0, rotation_angle_deg(rotation_x(20)), 1e-9, "deg")

# ---- fit_plane ------------------------------------------------------------------------------
checks.write()
checks.write("fit_plane: centroid and normal (the sign of the normal is arbitrary)")
angles = np.radians(np.arange(0, 360, 45))
circle_mm = np.column_stack([3 * np.cos(angles), 3 * np.sin(angles), np.full(8, 5.0)])   # radius 3 at z = 5
centroid_mm, normal = fit_plane(circle_mm)
checks.check("circle at z = 5: |n . (0,0,1)|", 1.0, abs(normal @ [0, 0, 1]), TOL)
checks.check("circle at z = 5: centroid", [0, 0, 5], centroid_mm, TOL, "mm")
# Points on the plane x + y + z = 6, whose normal is (1,1,1)/sqrt(3).
plane_points_mm = np.array([[1, 2, 3], [2, 1, 3], [2, 2, 2], [3, 0, 3], [3, 1, 2], [0, 3, 3], [0, 2, 4]], dtype=float)
centroid_mm, normal = fit_plane(plane_points_mm)
checks.check("plane x + y + z = 6: |n . (1,1,1)/sqrt(3)|", 1.0, abs(normal @ (np.ones(3) / np.sqrt(3))), TOL)
checks.check("plane x + y + z = 6: x + y + z of the centroid", 6.0, centroid_mm.sum(), TOL, "mm")

checks.finish("test_utilities")

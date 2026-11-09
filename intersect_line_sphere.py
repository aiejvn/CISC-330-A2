"""
intersect_line_sphere.py - where a line crosses a sphere (Task 3 utility).

The line is L(t) = P + t v with v normalized here, so t is a signed distance in mm along v
from P (t > 0 ahead of P, t < 0 behind it). The sphere has center C and radius R.

Geometry (report: "Tissue-core length", Fig. task3_geometry):
  1. The point of the line closest to C is at t_mid = (C - P) . v, the projection of C
     onto the line.
  2. Its distance from C is d, the distance of C from the line.
  3. C, the closest point and a point where the line meets the sphere form a right triangle
     with legs d and h and hypotenuse R, so by Pythagoras the half-chord is h = sqrt(R^2 - d^2).
  4. The line is inside the sphere from t_in = t_mid - h to t_out = t_mid + h.
  d > R: no real h, the line misses.  d = R: h = 0, the line just touches at t_mid.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize
from distance_point_line import distance_point_from_line

# d is computed in floating point and is almost never exactly R, so a tangent line is
# accepted when |d - R| is below this (mm). 1e-9 mm is far below any physical meaning.
TANGENT_TOL_mm = 1e-9


def intersect_line_sphere(P, v, C, R, tol=TANGENT_TOL_mm):
    """
    The t values (mm along unit v from P) where the line meets the sphere, sorted:
      []             the line misses the sphere (d > R)
      [t]            the line just touches it (|d - R| <= tol), at the closest point
      [t_in, t_out]  the line passes through it (d < R)
    """
    # Unit direction, so t is in mm.
    v = normalize(v)
    # Step 1: projection of C onto the line, the t of the closest point.
    t_mid = np.dot(np.asarray(C, dtype=float) - np.asarray(P, dtype=float), v)
    # Step 2: distance of C from the line.
    d = distance_point_from_line(C, P, v)
    # Miss: the closest point is already outside the sphere.
    if d > R + tol:
        return []
    # Tangent: the closest point lies on the sphere, the only common point.
    if abs(d - R) <= tol:
        return [t_mid]
    # Steps 3 and 4: half-chord by Pythagoras, then the entry and exit points.
    h = np.sqrt(R**2 - d**2)
    return [t_mid - h, t_mid + h]

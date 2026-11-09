"""
intersect_line_sphere.py - where a line crosses a sphere.

The line is L(t) = P + t v with v normalized here, so t is a distance (mm) along v from P,
negative behind P. The sphere has center C and radius R.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize
from distance_point_line import distance_point_from_line

TANGENT_TOL_mm = 1e-9


def intersect_line_sphere(P, v, C, R, tol=TANGENT_TOL_mm):
    """
    The t values where the line meets the sphere, sorted: [] if the line misses,
    [t] if it just touches (|d - R| <= tol), [t_in, t_out] if it passes through.
    The point of the line closest to C is at t_mid = (C - P) . v, a distance d from C.
    The line is inside the sphere for half a chord h = sqrt(R^2 - d^2) on either side of it.
    """
    v = normalize(v)
    t_mid = np.dot(np.asarray(C, dtype=float) - np.asarray(P, dtype=float), v)
    d = distance_point_from_line(C, P, v)
    if d > R + tol:
        return []
    if abs(d - R) <= tol:
        return [t_mid]
    h = np.sqrt(R**2 - d**2)
    return [t_mid - h, t_mid + h]

"""
distance_point_line.py - distance of a point from a line (Task 3 utility).

The line is L(t) = P + t v. Any point X splits, relative to P, into a part along the line
and a part perpendicular to it:

    X - P = ((X - P) . v) v  +  [ (X - P) - ((X - P) . v) v ]
            along the line      perpendicular to the line

The perpendicular part is the shortest way from the line to X, so its length is the
distance. This needs v of length 1 for the projection to be a length, so v is normalized
here. Task 3 uses it with X = the tumor center to get the targeting error (report:
"Targeting error", Fig. task3_geometry).

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize


def distance_point_from_line(point, P, v):
    """
    Shortest distance from `point` X to the line through P with direction v (mm in, mm out).
    v may have any length; only its direction is used.
    """
    # Unit direction, so that the dot product below is a length in mm.
    v = normalize(v)
    # Vector from the point on the line to X.
    to_point = np.asarray(point, dtype=float) - np.asarray(P, dtype=float)
    # Remove the part along the line; what is left is perpendicular to it.
    perpendicular = to_point - np.dot(to_point, v) * v
    return np.linalg.norm(perpendicular)

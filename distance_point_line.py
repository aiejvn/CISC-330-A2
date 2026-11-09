"""
distance_point_line.py - distance of a point from a line.

The line is L(t) = P + t v. The direction v need not have length 1; it is normalized here.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize


def distance_point_from_line(point, P, v):
    """
    Shortest distance from `point` X to the line through P with direction v.
    Remove from X - P its part along the unit direction; what is left is perpendicular
    to the line, and its length is the distance: |(X - P) - ((X - P) . v) v|.
    """
    v = normalize(v)
    to_point = np.asarray(point, dtype=float) - np.asarray(P, dtype=float)
    return np.linalg.norm(to_point - np.dot(to_point, v) * v)

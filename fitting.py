"""
fitting.py - least-squares plane fit, used by the tool-axis calibration.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np


def fit_plane(points_mm):
    """
    Least-squares plane through N >= 3 points (N, 3).
    Returns (centroid_mm, unit_normal). The plane passes through the centroid, and its
    normal is the direction in which the centered points spread least, i.e. the last
    right singular vector. The sign of the normal is arbitrary.
    """
    points_mm = np.asarray(points_mm, dtype=float)
    centroid_mm = points_mm.mean(axis=0)
    _, _, Vt = np.linalg.svd(points_mm - centroid_mm)
    return centroid_mm, Vt[-1]

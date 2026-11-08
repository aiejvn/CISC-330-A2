"""
frames.py - the marker frame built from the three markers.

Both tasks use it. In Task 1 the markers in each tracked pose give the transformation
from the marker frame to the tracker frame. In Task 2 the markers in F_mod and in F_ct
give the same marker frame twice. The markers must always be passed in the order
M1, M2, M3, otherwise a different frame comes out.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np
from geometry import normalize, frame_to_home


def frame_from_three_points(A, B, C):
    """
    Marker frame of the points A, B, C (each (3,), mm) given in some home frame:
        i = normalize(B - A),  k = normalize(i x (C - A)),  j = k x i,  O = (A + B + C) / 3
    Returns F_home_from_marker (4, 4).
    """
    A, B, C = (np.asarray(point, dtype=float) for point in (A, B, C))
    i = normalize(B - A)
    k = normalize(np.cross(i, C - A))
    j = np.cross(k, i)
    O = (A + B + C) / 3.0
    return frame_to_home(O, i, j, k)


def marker_frames(markers_mm):
    """Marker frame of every pose: markers_mm (n_poses, 3, 3) -> F_home_from_marker (n_poses, 4, 4)."""
    return np.array([frame_from_three_points(*markers_of_pose) for markers_of_pose in markers_mm])

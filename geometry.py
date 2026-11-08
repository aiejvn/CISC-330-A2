"""
geometry.py - basic vector and 4x4 transformation helpers.

Points are column vectors and a 4x4 matrix acts from the left. F_a_from_b is the
transformation from frame b to frame a. Functions take one point (3,) or many (N, 3).

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np


def normalize(vector):
    """Unit vector in the direction of `vector`."""
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def translation(d):
    """4x4 translation by d."""
    T = np.eye(4)
    T[:3, 3] = d
    return T


def rotation_x(angle_deg):
    """4x4 rotation about x."""
    a = np.radians(angle_deg)
    R = np.eye(4)
    R[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return R


def rotation_y(angle_deg):
    """4x4 rotation about y."""
    a = np.radians(angle_deg)
    R = np.eye(4)
    R[0, 0], R[0, 2], R[2, 0], R[2, 2] = np.cos(a), np.sin(a), -np.sin(a), np.cos(a)
    return R


def rotation_z(angle_deg):
    """4x4 rotation about z."""
    a = np.radians(angle_deg)
    R = np.eye(4)
    R[0:2, 0:2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return R


def frame_to_home(O_e, e1, e2, e3):
    """F_home_from_e for a frame with origin O_e and axes e1, e2, e3 given in home coordinates."""
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2] = e1, e2, e3
    F[:3, 3] = O_e
    return F


def invert_rigid(F):
    """Inverse of a rigid transformation: [R^T, -R^T t]."""
    R, t = F[:3, :3], F[:3, 3]
    F_inv = np.eye(4)
    F_inv[:3, :3] = R.T
    F_inv[:3, 3] = -R.T @ t
    return F_inv


def apply_transform(F, points):
    """Apply a 4x4 transformation to one point (3,) or many points (N, 3)."""
    points = np.asarray(points, dtype=float)
    return points @ F[:3, :3].T + F[:3, 3]


def transform_direction(F, directions):
    """
    Apply a 4x4 transformation to one direction (3,) or many (N, 3).
    A direction has no position, so only the rotation part of F acts on it.
    """
    return np.asarray(directions, dtype=float) @ F[:3, :3].T


def is_rigid(F, tol=1e-9):
    """True if the rotation part of F is orthonormal and right-handed."""
    R = F[:3, :3]
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol


def angle_between_deg(u, v):
    """
    Angle between two vectors in degrees. atan2 of the cross and dot products stays
    accurate for nearly parallel vectors, where arccos of the dot product does not.
    """
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


def rotation_angle_deg(R):
    """Angle of a rotation matrix (3x3, or the rotation part of a 4x4), in degrees."""
    R = np.asarray(R, dtype=float)[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))

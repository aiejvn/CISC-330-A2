"""
noise.py - Gaussian marker-localization error and error statistics.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import numpy as np


def add_marker_noise(points_mm, sigma_mm, rng):
    """
    Noisy copy of marker positions of any shape (..., 3). Every coordinate gets its own
    zero-mean Gaussian error with standard deviation sigma_mm. rng is a numpy random
    generator, so a fixed seed repeats the same noise.
    """
    points_mm = np.asarray(points_mm, dtype=float)
    return points_mm + rng.normal(0.0, sigma_mm, size=points_mm.shape)


def error_statistics(errors):
    """Mean and 95th percentile of a 1-D array of errors."""
    errors = np.asarray(errors, dtype=float)
    return {"mean": errors.mean(), "p95": np.percentile(errors, 95)}

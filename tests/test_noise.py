"""
Test of the marker noise level: 95% of the 3-D marker errors must be shorter than 1.0 mm.

With sigma on each of x, y, z, |error|^2 / sigma^2 follows a chi-square distribution
with 3 degrees of freedom, so sigma = 1.0 mm / sqrt(7.8147) = 0.358 mm.

Run from the A2 folder:   python3 tests/test_noise.py

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from checks import CheckList
from noise import add_marker_noise

N_DRAWS = 200_000


def chi_square_3dof_cdf(x):
    """P(chi-square with 3 degrees of freedom < x) = erf(sqrt(x/2)) - sqrt(2x/pi) exp(-x/2)."""
    return math.erf(math.sqrt(x / 2)) - math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


checks = CheckList("Noise test: 95% of the 3-D marker errors shorter than 1.0 mm")

checks.write()
checks.write("1. Choice of sigma")
checks.check("chi-square (3 dof) CDF at 7.814728", 0.95, chi_square_3dof_cdf(config.CHI2_3DOF_95), 1e-9)
checks.check("sigma = 1.0 mm / sqrt(7.814728)", 0.357720, config.MARKER_SIGMA_mm, 1e-6, "mm")

checks.write()
checks.write(f"2. Simulation: {N_DRAWS} marker errors with sigma = {config.MARKER_SIGMA_mm:.4f} mm")
rng = np.random.default_rng(config.RANDOM_SEED)
errors_mm = add_marker_noise(np.zeros((N_DRAWS, 3)), config.MARKER_SIGMA_mm, rng)
checks.check("fraction of 3-D errors < 1.0 mm", 0.95, np.mean(np.linalg.norm(errors_mm, axis=1) < 1.0), 0.005)
checks.check("mean error on x, y, z", [0, 0, 0], errors_mm.mean(axis=0), 0.005, "mm")
checks.check("standard deviation on x, y, z / sigma", [1, 1, 1], errors_mm.std(axis=0) / config.MARKER_SIGMA_mm, 0.01)

checks.write()
checks.write("3. For comparison: sigma = 0.5 mm, from the one-dimensional rule 2 sigma = 1.0 mm")
errors_mm = add_marker_noise(np.zeros((N_DRAWS, 3)), 0.5, rng)
checks.check("fraction of 3-D errors < 1.0 mm, sigma 0.5", chi_square_3dof_cdf((1.0 / 0.5) ** 2),
             np.mean(np.linalg.norm(errors_mm, axis=1) < 1.0), 0.005, "", "only about 74% in 3-D")

checks.finish("test_noise")

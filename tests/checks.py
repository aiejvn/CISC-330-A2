"""
checks.py - a small PASS/FAIL helper shared by the test files.

Each check prints the predicted value, the computed value, their difference and PASS or
FAIL. finish() saves the printout to outputs/<name>.txt and exits with status 1 if any
check failed.

AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def value_text(value):
    """A number or array as short one-line text (6 significant digits, round-off below 1e-12 shown as 0)."""
    array = np.asarray(value, dtype=float)
    array = np.where(np.abs(array) < 1e-12, 0.0, array)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    text = np.array2string(array, separator=", ", formatter={"float_kind": lambda v: f"{v:.6g}"},
                           max_line_width=10_000)
    return " ".join(text.split())


class CheckList:
    def __init__(self, title):
        self.lines = []
        self.n_checks = 0
        self.n_failed = 0
        self.write(title)
        self.write("=" * len(title))

    def write(self, text=""):
        print(text)
        self.lines.append(text)

    def _record(self, passed, line):
        self.n_checks += 1
        self.n_failed += not passed
        self.write(line)

    def check(self, label, predicted, computed, tol, unit="", note=""):
        """PASS if the largest absolute difference between computed and predicted is at most tol."""
        difference = float(np.max(np.abs(np.asarray(computed, float) - np.asarray(predicted, float))))
        passed = difference <= tol
        line = (f"  {label:<46s} predicted {value_text(predicted):<26s} computed {value_text(computed):<26s} "
                f"difference {difference:.1e} {unit:<4s}(tol {tol:.0e})  {'PASS' if passed else 'FAIL'}")
        self._record(passed, line + (f"  | {note}" if note else ""))
        return passed

    def check_true(self, label, passed, note=""):
        """A yes/no check."""
        line = f"  {label:<46s} {'PASS' if passed else 'FAIL'}"
        self._record(bool(passed), line + (f"  | {note}" if note else ""))
        return passed

    def finish(self, output_name):
        self.write()
        self.write(f"Number of checks: {self.n_checks}")
        self.write(f"Number of failed checks: {self.n_failed}  ->  "
                   f"{'ALL CHECKS PASSED' if self.n_failed == 0 else 'SOME CHECKS FAILED'}")
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, output_name + ".txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")
        if self.n_failed:
            sys.exit(1)

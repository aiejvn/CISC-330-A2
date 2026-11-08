#!/bin/bash
# Runs all tests and the Task 1 noisy calibration (about 1 minute):   bash run_tests.sh
# AI use: written with help from Claude (Anthropic), Nov 2026; reviewed and tested by the group.
cd "$(dirname "$0")"
status=0
for script in tests/test_utilities.py tests/test_noise.py tests/test_task1_calibration.py \
              tests/test_task2_registration.py run_task1_calibration.py; do
    echo "=== $script"; python3 "$script" || status=1
done
exit $status

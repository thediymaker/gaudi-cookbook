# SPDX-License-Identifier: Apache-2.0
"""Make glm5_hpu and the converters importable from a checkout (no install needed for the CPU tests)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "glm5_hpu"))
sys.path.insert(0, os.path.join(HERE, "..", "convert"))

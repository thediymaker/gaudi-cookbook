# SPDX-License-Identifier: Apache-2.0
"""Make the port's plugin package importable from a checkout (no install needed for the CPU tests)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "beam_hpu"))

# SPDX-License-Identifier: Apache-2.0
"""Make the port's plugin package importable from a checkout (no install needed for the CPU tests), and skip the
tests that need torch when it is not installed."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml4_hpu"))

collect_ignore = ["build_gates.py", "check_spec_grammar.py", "hpu"]
if importlib.util.find_spec("torch") is None:
    collect_ignore += ["test_k1_cpu.py", "test_k2_cpu.py"]

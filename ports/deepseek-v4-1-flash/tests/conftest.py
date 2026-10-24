# SPDX-License-Identifier: Apache-2.0
"""Puts the port's plugin package and this directory on sys.path. Tests that need torch skip without it."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "dsv41_hpu"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

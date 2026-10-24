# SPDX-License-Identifier: Apache-2.0
"""Puts convert/ and patches/ on sys.path. Tests that need torch or release sources skip without them."""

import sys
from pathlib import Path

PORT = Path(__file__).resolve().parent.parent
for p in (PORT / "convert", PORT / "patches"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

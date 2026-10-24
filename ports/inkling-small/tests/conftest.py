# SPDX-License-Identifier: Apache-2.0
"""Make the port's plugin package importable without installing it."""

import sys
from pathlib import Path

PORT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORT / "inkling_hpu"))

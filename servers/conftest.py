# SPDX-License-Identifier: Apache-2.0
"""Put servers/common on the import path for every server's CPU tests, as the Dockerfiles' PYTHONPATH does."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

# SPDX-License-Identifier: Apache-2.0
"""Import a server's modules in its CPU tests. The servers reuse module names (server, engine), so one pytest run
over servers/ must drop the previous server's modules before importing the next one's."""

import importlib
import os
import sys

NAMES = ("server", "engine", "media", "model", "t3", "s3gen", "speech_stream")


def load(test_file: str, *names: str):
    here = os.path.dirname(os.path.dirname(os.path.abspath(test_file)))
    for n in NAMES:
        sys.modules.pop(n, None)
    sys.path.insert(0, here)
    try:
        return [importlib.import_module(n) for n in names]
    finally:
        sys.path.remove(here)

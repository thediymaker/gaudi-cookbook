# SPDX-License-Identifier: Apache-2.0
"""Shared by every port's tests. A port targets one release (the `vllm_gaudi` of the recipes that name it in
`gaudi.port`). When GAUDI_FIXES_PROFILE names a different release, the port's tests are skipped: its patches are
written against its own release's sources."""

import os
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _targets():
    out = {}
    for y in (HERE.parent / "recipes").glob("*/*/recipe.yaml"):
        text = y.read_text()
        port = re.search(r"^  port: \"?([\w.-]+)\"?\s*(?:#.*)?$", text, re.M)
        release = re.search(r"^    vllm_gaudi: \"(?:releases/)?(v[\d.]+)\"", text, re.M)
        if port and release:
            out[port.group(1)] = release.group(1)
    return out


def pytest_collection_modifyitems(config, items):
    profile = os.environ.get("GAUDI_FIXES_PROFILE")
    if not profile:
        return
    targets = _targets()
    for item in items:
        try:
            port = Path(str(item.fspath)).resolve().relative_to(HERE).parts[0]
        except ValueError:
            continue
        target = targets.get(port)
        if target and target != profile:
            item.add_marker(pytest.mark.skip(reason=f"{port} targets {target}, GAUDI_FIXES_PROFILE is {profile}"))

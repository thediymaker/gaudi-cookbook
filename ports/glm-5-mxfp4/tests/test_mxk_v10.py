# SPDX-License-Identifier: Apache-2.0
"""glm5_hpu.mxfp4_moe on CPU: which v10 decode path a library serves, and the per-call fallback."""

from glm5_hpu import mxfp4_moe as m

V8 = {"mxk_w13x", "mxk_w2x", "mxk_deq"}
STACK3 = V8 | {"mxk_w13xn", "mxk_w2xc", "mxk_w2gp", "mxk_w2c"}


def test_pick_v10_path():
    assert m.pick_v10_path(STACK3, "auto") == "stack3"
    assert m.pick_v10_path(STACK3, "w2xc") == "stack2"
    assert m.pick_v10_path(V8 | {"mxk_w13xn", "mxk_w2xc"}, "auto") == "stack2"
    assert m.pick_v10_path(V8, "auto") is None
    assert m.pick_v10_path(V8 | {"mxk_w2gp", "mxk_w2c"}, "auto") is None  # no mxk_w13xn


def test_v10_call_path():
    assert m.v10_call_path("stack3", 16, 8, 1024) == "stack3"
    assert m.v10_call_path("stack3", 16, 8, 3072) == "stack3"
    assert m.v10_call_path("stack3", 16, 8, 3584) == "stack2"
    assert m.v10_call_path("stack3", 65, 8, 1024) == "stack2"
    assert m.v10_call_path("stack2", 1, 8, 1024) == "stack2"


def test_group_limits_match_the_kernel_glue():
    # src/mxk_tpc_lib.cpp: mxk_w2gp / mxk_w2c refuse Q > 512 and I > 3072; gmax 2 is the setting that was gated
    assert (m.GROUP_Q_MAX, m.GROUP_I_MAX, m.GROUP_GMAX) == (512, 3072, 2)

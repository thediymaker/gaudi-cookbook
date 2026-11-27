# SPDX-License-Identifier: Apache-2.0
"""Which Muse Spark MoE layers the MXFP4 kernels accept (CPU, no torch, no device).

mxk_tpc_lib.cpp refuses I > 3072 in both kernels of the muse variant (instW13 and instW2), so any layout that
leaves a rank more intermediate width than that must stay on the stock op.
"""

from muse_spark_hpu import mxfp4_kernels as mxk


def test_recipe_layout_runs_on_the_kernels():
    # EP 2 x TP 4: 128 local experts, Hm 4096, I 1024 per rank, top-8
    assert mxk.unsupported(128, 4096, 1024, 8) is None


def test_whole_intermediate_on_one_rank_stays_on_the_stock_op():
    # MUSE_SPARK_MOE_TP_SPLIT=1 (EP 8): each rank keeps the whole 4096 intermediate
    why = mxk.unsupported(32, 4096, 4096, 8)
    assert why and "4096 > 3072" in why


def test_intermediate_cap_is_inclusive():
    assert mxk.unsupported(128, 4096, 3072, 8) is None
    assert mxk.unsupported(128, 4096, 3584, 8)


def test_other_limits():
    assert mxk.unsupported(129, 4096, 1024, 8)  # more than 128 local experts
    assert mxk.unsupported(128, 4096, 1000, 8)  # not a multiple of 512
    assert mxk.unsupported(128, 6656, 1024, 8)  # w13 input wider than 6144
    assert mxk.unsupported(128, 4096, 1024, 64)  # MAX_TOKENS x top-k over 128 routes

# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 and GLM-5.3 (MXFP4) on Intel Gaudi 2: the runtime code of ports/glm-5-mxfp4.

vLLM already has the model (GlmMoeDsaForCausalLM). This package holds what the port adds, and the port's patches
call into it from the vllm-gaudi modules that run at the right moment (before the device opens, after the weights
load, inside the MLA wrapper), so it needs no plugin entry point:

* :mod:`glm5_hpu.mxfp4_moe`: the MXFP4 MoE decode kernels and the prefill unpack (GLM_MXK, GLM_MXK_PREFILL)
* :mod:`glm5_hpu.dsa`: DeepSeek sparse attention on the HPU MLA path (GLM_DSA_SPARSE)
* :mod:`glm5_hpu.moe_layout`: EP 4 x TP 2 routed experts and a static expert placement (GLM_MOE_TP, GLM_MOE_PLACEMENT)
* :mod:`glm5_hpu.decode_lean`: decode-step levers (GLM_ROUTER_FAST, GLM_SPEC_DEVICE_REJECT, GLM_EARLY_DRAFT)

Every switch is off by default, and off runs the stock code path.
"""

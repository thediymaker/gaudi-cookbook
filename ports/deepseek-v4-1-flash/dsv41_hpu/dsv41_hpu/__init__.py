# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash for vLLM on Intel Gaudi 2 (vllm-gaudi v0.30.0).

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: dsv41_hpu`): registers the HPU model for
architecture DeepseekV41ForCausalLM and the `dsv41_hpu_mxfp4` quant config the converted checkpoint declares,
corrects the frontend's reasoning-effort aliases (frontend.py) and, with DSV41_REPLAY_KEEP > 0, caps prefix-cache
hits for decoder replay (replay.py; the scheduler runs in this process). The HPU runner hook is installed by the
model constructor, inside the worker process, before any batch is prepared.
"""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model("DeepseekV41ForCausalLM", "dsv41_hpu.vllm_model:DeepseekV41HPUForCausalLM")
    from . import frontend, quant  # noqa: F401  (quant registers dsv41_hpu_mxfp4)

    frontend.install()
    from . import replay

    replay.install_hit_cap()

"""Mistral Large 4 (AMD Quark MXFP4) for vLLM on Intel Gaudi (vllm-gaudi v0.30).

Loaded as a vLLM general plugin (entry point `vllm.general_plugins: mistral_large4_hpu`), in every process:
API server, engine core and workers. register():
  - checks the frontend: mistral-common >= 1.13.0 with the low / medium / high efforts (frontend.py; fails
    closed);
  - checks that the vendored PR #64534 files are in the vLLM tree (the params.json adapter with the `dsa`
    branch); without them registration fails loudly;
  - points architecture MistralLarge4ForCausalLM at our HPU model (lazily, by module path, so registration
    never imports HPU code). v0.30's Pixtral wrapper builds its language model through the registry, so the
    same name serves the vision phase;
  - installs the text-only switch on the params.json adapter (config.py; ML4_VISION=1 turns it off);
  - registers the `mistral_large4_hpu_mxfp4` quant config, which claims the AMD Quark checkpoint (quant.py);
  - registers the load format (loader.py), when the package has one;
  - installs the prefix-cache hit cap (prefix.py: ML4_HIT_KEEP / ML4_HIT_ALIGN, both off by default; only the
    engine process schedules, the patch is inert elsewhere).

The PR files are installed into the vLLM tree by the port's install.sh (patches/install_pr_files.py). The HPU runner fixes are installed by the model constructor, inside the worker process.

Serve: --config-format mistral --tokenizer-mode mistral --load-format mistral.
"""

MODEL_ARCH = "MistralLarge4ForCausalLM"
MODEL_CLASS = "mistral_large4_hpu.vllm_model:ML4ForCausalLM"


def register() -> None:
    import importlib.util

    from vllm import ModelRegistry

    from . import frontend

    frontend.install()
    from . import asyncspec

    asyncspec.install_process()  # ML4_ASYNC_SPEC=1 only: engine async scheduling with DSpark (prototype)
    import vllm.transformers_utils.configs.mistral as C

    if not hasattr(C, "_remap_mistral_large_4_args"):
        raise ImportError(
            "mistral_large4_hpu: vLLM PR #64534 files missing from the vLLM tree; run patches/install_pr_files.py"
        )

    ModelRegistry.register_model(MODEL_ARCH, MODEL_CLASS)
    from . import config

    config.install()
    from . import quant  # noqa: F401  (registers mistral_large4_hpu_mxfp4)
    from . import prefix

    prefix.install_hit_cap()  # ML4_HIT_KEEP / ML4_HIT_ALIGN (engine process; off by default)
    if importlib.util.find_spec(f"{__name__}.loader") is not None:
        try:  # registers the load format (opt-in by --load-format; the default stays "auto")
            from . import loader  # noqa: F401
        except Exception as e:  # never let the optional loader break registration of the rest
            import logging

            logging.getLogger("vllm.mistral_large4_hpu").warning("mistral_large4_hpu: load format unavailable: %r", e)

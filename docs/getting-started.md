# Getting started

Everything in this cookbook was run on **Intel Gaudi 2** (HL-225, 96 GB HBM per card,
8 cards per node) with the components below. Gaudi 3 is not covered yet.

## Software baseline

| Component | Version |
| --- | --- |
| Gaudi software (driver, firmware, container base) | 1.24.1 |
| Container base image | `vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0` |
| PyTorch | Habana PyTorch 2.11 fork (ships in the base image) |
| vllm-gaudi | `v0.26.0` release, or the `releases/v0.30.0` branch |
| vLLM | the commit each vllm-gaudi release tracks (`VLLM_STABLE_COMMIT`) |
| Execution mode | lazy mode (`PT_HPU_LAZY_MODE=1`) unless a recipe says otherwise |

!!! warning "Match the host driver"
    The container's Gaudi software release must match the host driver. A 1.24.2
    container on a 1.24.1 host is not supported. Use the `pytorch-installer` base,
    not `pytorch-upstream-installer`: upstream PyTorch does not support lazy mode.

## Building an image

vllm-gaudi publishes Dockerfiles under
[`.cd/`](https://github.com/vllm-project/vllm-gaudi/tree/main/.cd). The pattern used for
every recipe here:

1. Start from the base image above.
2. Clone vllm-gaudi at a release tag (pin by commit) and read its `VLLM_STABLE_COMMIT`.
3. Clone vLLM at that commit and install it with `VLLM_TARGET_DEVICE=empty`, after
   removing `torch` from vLLM's build requirements so the Habana PyTorch fork is kept.
4. On vLLM releases that carry the `GraphCaptureOutput.get_runtime_env` workaround for
   torch below 2.12 in `vllm/env_override.py`, comment that block out: the Habana
   PyTorch 2.11 fork does not have the class it patches. Keep (or stub) the
   `_apply_constrain_to_fx_strides_patch` function that other modules import.
5. Install vllm-gaudi with `VLLM_TARGET_DEVICE=hpu pip install -e . --no-build-isolation`.
6. Check that `torch.__version__` has no `+cpu` or `+cu` suffix after the install, and
   that `import habana_frameworks.torch` still works.

## Running a recipe

Each recipe gives a `vllm serve` command and the environment it needs. Inside a
container with Gaudi devices visible:

```bash
docker run -it --rm --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice --ipc=host --net=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  <your-vllm-gaudi-image> \
  vllm serve <model> <recipe flags>
```

On Kubernetes, see [Kubernetes](kubernetes/index.md) for the pod settings Gaudi
needs (`runtimeClassName: habana`, `hostIPC: true`, hugepages, device requests).

## Fixes

Some recipes list **required fixes**. These are small runtime patches in
[`fixes/`](fixes/index.md) for bugs in the released images, each tied to an upstream
pull request. They run before `vllm serve` and refuse to start the server if the
image does not match what they expect.

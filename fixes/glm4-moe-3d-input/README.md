# glm4-moe-3d-input

`Glm4MoE.forward` accepts the 3-D hidden states the HPU runner passes, so GLM-4.5 family MoE
models generate instead of stopping the engine on the first token.

| | |
| --- | --- |
| Affects | models served through vLLM's `glm4_moe.py` MoE block: GLM-4.5, GLM-4.5-Air, GLM-4.5V, GLM-4.6 |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/model_executor/models/glm4_moe.py` |
| Upstream | not filed yet |
| Test | `test_glm4_moe_3d_input.py` (CPU, needs a release source tree in `GAUDI_FIXES_ROOT`) |

## Problem

The MoE block opens with `num_tokens, hidden_dim = hidden_states.shape`, which assumes
`[tokens, hidden]`. The vllm-gaudi runner drives models with `[batch, seq, hidden]`.

## Symptom

The server starts and `/health` answers, then the first request fails with
`ValueError: too many values to unpack (expected 2)` in a TP worker and the engine stops
(`EngineDeadError`, HTTP 500).

## Who is affected

Every model whose MoE layers come from `glm4_moe.py`, served on HPU.

## The fix

The block flattens to `[tokens, hidden]` on entry and restores the caller's shape on exit. For 2-D
input it is a no-op.

## Verify

```bash
python3 fixes/glm4-moe-3d-input/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/glm4-moe-3d-input
```

End to end: GLM-4.5V-FP8 on two Gaudi 2 cards, `releases/v0.30.0`: the stock image stops on the first
chat request with the error above; with the fix the model answers.

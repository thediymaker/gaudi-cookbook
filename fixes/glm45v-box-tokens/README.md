# glm45v-box-tokens

The `glm45` reasoning parser consumes GLM-4.5V's `<|begin_of_box|>` and `<|end_of_box|>` answer
markers instead of passing them to the client.

| | |
| --- | --- |
| Affects | GLM-4.5V with `--reasoning-parser glm45` |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/parser/glm47_moe.py` |
| Upstream | not filed yet |
| Test | `test_glm45v_box_tokens.py` (CPU, needs a release source tree in `GAUDI_FIXES_ROOT`) |

## Problem

GLM-4.5V wraps short final answers in `<|begin_of_box|>` ... `<|end_of_box|>`. vLLM maps
`--reasoning-parser glm45` onto the unified GLM-4.7 parser, which has no rule for these markers.
`skip_special_tokens` does not remove them: with a reasoning parser active the detokenized text keeps
special tokens so the parser can match `<think>` and `</think>`.

## Symptom

`content` such as `"<|begin_of_box|>Paris<|end_of_box|>"`, streamed and not streamed, mostly on
short factual answers.

## Who is affected

GLM-4.5V served with the `glm45` (or `glm47`) reasoning parser. Models that never emit the markers,
such as GLM-4.7, are unaffected.

## The fix

The two markers become parser terminals with transitions that emit nothing, in the content and
reasoning states, the same way the parser already swallows a stray `</think>`.

## Verify

```bash
python3 fixes/glm45v-box-tokens/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
GAUDI_FIXES_ROOT=<site-packages of a release> pytest fixes/glm45v-box-tokens
```

End to end: short factual questions ("What is the capital of France?") to GLM-4.5V with
`--reasoning-parser glm45`, streamed and not streamed; `content` must not contain either marker.
GLM-4.5V-FP8 on two Gaudi 2 cards, `releases/v0.30.0`: 4 of 4 answers carried the markers without
the fix (streamed too), 0 of 4 with it.

# structured-outputs-auto-whitespace

Source: [`fixes/structured-outputs-auto-whitespace/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/structured-outputs-auto-whitespace)

Allow `disable_any_whitespace` with the default `auto` structured-output backend, and make the
outlines backend honour it.

| | |
| --- | --- |
| Affects | servers that want compact JSON from structured outputs (`disable_any_whitespace`) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/config/structured_outputs.py`, `vllm/v1/structured_output/backend_outlines.py` |
| Upstream | [vllm#63430](https://github.com/vllm-project/vllm/pull/63430) (open) |
| Test | none |

## Problem

`--structured-outputs-config '{"disable_any_whitespace": true}'` is rejected at startup with
`disable_any_whitespace is only supported for xgrammar and guidance backends` unless the backend
is named explicitly, because the validator checks the backend name and the default is `auto`.
Naming `xgrammar` avoids the error but loses the fallback that `auto` provides: schemas xgrammar
cannot compile (for example `multipleOf`, `uniqueItems`, `patternProperties`) then fail with
HTTP 400 instead of falling back to guidance.

## Symptom

The server refuses to start with the flag, or, with `backend: xgrammar`, some JSON schemas that
work under `auto` start returning 400. Without the flag, JSON grammars allow arbitrary
whitespace, and some models (reasoning models after a forced tool call in particular) then emit
whitespace until `max_tokens`.

## Who is affected

Anyone who wants compact JSON output while keeping the default backend selection.

## The fix

xgrammar and guidance already read the flag from the engine config. The fix makes outlines pass
an empty whitespace pattern to `outlines_core` when the flag is set, so every backend that `auto`
can choose honours it, and relaxes the validator to reject only `lm-format-enforcer`. This matches
the upstream pull request.

## Verify

```bash
python3 fixes/structured-outputs-auto-whitespace/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

Start the server with `--structured-outputs-config '{"disable_any_whitespace": true}'` and no
backend: it must start, and a `response_format` JSON schema request returns compact JSON.

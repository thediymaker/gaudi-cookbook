# gptoss-harmony-parse-recovery

Source: [`fixes/gptoss-harmony-parse-recovery/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/gptoss-harmony-parse-recovery)

Keep a malformed gpt-oss message header from failing the whole request with HTTP 500, and keep
the tool call it was meant to carry.

| | |
| --- | --- |
| Affects | gpt-oss (20B and 120B) with the harmony parser, mostly requests with tools |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/parser/harmony.py` |
| Upstream | [vllm#59387](https://github.com/vllm-project/vllm/pull/59387) (open); vLLM main drops the bad token, without the splice |
| Test | none |

## Problem

gpt-oss sometimes starts a tool-call header, abandons it and starts a new one. The
`openai_harmony` `StreamableParser` is still inside the first header when the restart arrives
and raises `HarmonyError` at the second header's `<|message|>`. `HarmonyParser.process_chunk`
feeds tokens to the parser without error handling, so the error escapes to the serving layer.

## Symptom

The request fails with HTTP 500 and an error such as
`unexpected tokens remaining in message header: Some("to=functions.get_weather")`. It is
intermittent and most common with a system prompt and several tools.

## Who is affected

Anyone serving gpt-oss with tool calling, streaming or not.

## The fix

Each token goes through a guarded helper. On `HarmonyError` it first splices the message: the
recipient (`to=...`) from the abandoned header plus the payload of the restarted one, re-parsed
with a fresh parser, so the tool call survives. Resyncing at the restart instead would return a
recipient-less commentary message, prose where a tool call was meant. If the splice does not
parse, the token is dropped and parsing continues, and `flush()` returns the raw text for what
cannot be parsed. A rolling window of the last 512 tokens is kept across message boundaries,
because the abandoned header ends with `<|end|>`, which completes a message and clears the
per-message buffer before the error is raised. Each recovery logs a warning.

The script also applies on top of vLLM main's drop-token guard.

## Verify

```bash
python3 fixes/gptoss-harmony-parse-recovery/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: send a few hundred chat requests with a system prompt and three or more tools. Without
the fix some return HTTP 500; with it they return tool calls, and the server log shows
"malformed harmony header ... repaired by splicing the recipient".

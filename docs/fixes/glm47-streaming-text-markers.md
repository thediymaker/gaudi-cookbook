# glm47-streaming-text-markers

Source: [`fixes/glm47-streaming-text-markers/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/glm47-streaming-text-markers)

Make the `glm47` tool parser recognise tool-call markers written as plain text when streaming, so
streaming and non-streaming responses agree.

| | |
| --- | --- |
| Affects | GLM-4.7 and GLM-5 family models served with `--tool-call-parser glm47`, `stream: true`, forced tool calls |
| Releases | v0.26.0, v0.30.0 (one script, two source shapes) |
| Patches | `vllm/parser/glm47_moe.py` (checks `vllm/parser/engine/*.py`) |
| Upstream | not filed yet (the engine behaviour is unchanged on vLLM main) |
| Test | none |

## Problem

The GLM tokenizer has added tokens for `<tool_call>` and `</tool_call>`, but the model can also
write them as ordinary BPE pieces, and the byte-level structural tag used for forced tool calls
accepts both spellings. vLLM's streaming parser engine treats a terminal that also has a token-id
form as content once the stream has carried token ids, so in streaming the markers are recognised
only by id. The non-streaming path parses text, so it sees the same text as markers.

## Symptom

With `stream: true` and `tool_choice: "required"` or a named tool, some responses carry the whole
call as raw text in `delta.content`
(`<tool_call>name\n<arg_key>...</arg_key>\n<arg_value>...</arg_value>\n</tool_call>`), with no
`tool_calls` and `finish_reason: "stop"`. The same request with `stream: false` returns a proper
tool call. A second, rarer symptom: a call the model closes with a malformed value streams
arguments that are not valid JSON.

## Who is affected

Only deployments using the `glm47` parser. Other parsers are not touched, so this fix is not part
of the default profile; recipes for GLM models list it.

## The fix

For the `glm47` parser only, per parser instance and only in streaming:

- a plain-text `<tool_call>` opens a call after reasoning has ended when the request carries a
  structural tag, which means the grammar forces a call there; free-form chat and text inside
  reasoning are unchanged;
- a plain-text `</tool_call>` closes an open call;
- `<think>` and `</think>` stay id-only, since their plain spelling appears in prose;
- if the model closes a call while its last value is unterminated, the streamed arguments are
  closed as valid JSON. A call cut off by `max_tokens` is streamed exactly as before.

The override reproduces the engine's token loop, so the script checks the exact engine code of
each release line and refuses to patch anything else.

## Verify

```bash
python3 fixes/glm47-streaming-text-markers/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: send the same forced-tool-call request many times with `stream: true` and with
`stream: false`; every streamed response should carry `tool_calls` like the non-streamed one.

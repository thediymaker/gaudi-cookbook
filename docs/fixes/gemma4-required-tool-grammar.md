# gemma4-required-tool-grammar

Source: [`fixes/gemma4-required-tool-grammar/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/gemma4-required-tool-grammar)

Make `tool_choice: "required"` and named tool choice produce a tool call with the `gemma4` tool
parser.

| | |
| --- | --- |
| Affects | Gemma 4 models served with `--tool-call-parser gemma4` and forced tool choice |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/tool_parsers/gemma4_engine_tool_parser.py`, `vllm/tool_parsers/streaming.py`, `vllm/parser/abstract_parser.py` |
| Upstream | related issue [vllm#53363](https://github.com/vllm-project/vllm/issues/53363); fix not filed (vLLM main still skips the grammar) |
| Test | none (covered by the end-to-end check below) |

## Problem

The engine-based `gemma4` tool parser skips structured outputs for required and named tool choice,
so that the model can write its native `<|tool_call>call:...` syntax. With no grammar nothing
forces a call, and the parser also declares `supports_required_and_named = False`, so the serving
layer treats the request like `auto`.

## Symptom

A request with `tool_choice: "required"` or a named function returns prose and no `tool_calls`,
some of the time. Clients that rely on forced calls (agents, extraction pipelines) fail or loop.

## Who is affected

Anyone serving Gemma 4 with the `gemma4` tool parser who sends forced tool choice. `auto` tool
choice is unchanged.

## The fix

- `adjust_request` falls through to the base `ToolParser.adjust_request` for required and named
  tool choice, which constrains the output to the tools' JSON schema, as the generic parsers do.
- The parser declares `supports_required_and_named = True`, so the serving layer parses that JSON
  into tool calls, streaming and non-streaming.
- `extract_required_tool_call_streaming` treats an `IndexError` from `partial_json_parser` like
  `MalformedJSON` (not parseable yet), instead of ending the stream.
- Engine-based streams pass the tool extractor only the newest delta. For this parser only, the
  required/named streaming path keeps the accumulated text on the stream state and passes the full
  text to the JSON extractors. Other parsers are not affected.

Not for speculative decoding: grammar-constrained tool calls under speculation are a separate
open problem on HPU.

## Verify

```bash
python3 fixes/gemma4-required-tool-grammar/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: send the same chat request with one tool and `tool_choice: "required"` many times at
temperature 0.7, streaming and non-streaming. With the fix every response has a `tool_calls` entry
whose arguments are valid JSON for the tool's schema.

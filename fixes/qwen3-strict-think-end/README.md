# qwen3-strict-think-end

End Qwen3 reasoning only at `</think>`, so a tool-call example the model quotes while thinking is
not returned as a real tool call. Opt-in.

| | |
| --- | --- |
| Affects | Qwen3-family models served with `--reasoning-parser qwen3` and tool calling, for example Qwen3.6-27B |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/parser/qwen3.py` |
| Upstream | [vllm#59821](https://github.com/vllm-project/vllm/pull/59821) (open), issue [vllm#58147](https://github.com/vllm-project/vllm/issues/58147) |
| Test | none (parser state-table change; covered by the end-to-end check below) |

## Problem

vLLM's engine-based Qwen3 parser treats a `<tool_call>` that appears while the model is still
reasoning as an implicit end of reasoning: its grammar has a `(REASONING, TOOL_START)` transition
straight into a tool call. On the vLLM commit v0.26.0 tracks, `Qwen3Parser.is_reasoning_end` also
returns True on an unpaired `<tool_call>` token. This was added for models that call tools without
closing `</think>`.

## Symptom

While thinking, a model sometimes quotes the tool-format example from its chat template
(`<tool_call><function=example_function_name>...`). The parser ends the reasoning there and
returns the quoted example as a tool call, followed by the model's real call. Clients see an extra
call to a function that does not exist, or a duplicated call.

## Who is affected

Qwen3-family models that always close `</think>` before calling a tool and whose reasoning can
mention the tool-call format. Models that rely on the implicit end (they call tools without
closing `</think>`) need the stock behaviour, which is why this fix is opt-in.

## The fix

The implicit-end transition is removed from the parser's state table, so only `</think>` ends the
reasoning. On v0.26.0 the separate unpaired-`<tool_call>` check in `is_reasoning_end` is skipped as
well; on v0.30.0 that check is derived from the state table and needs no second edit. Applying the
fix enables the behaviour; there is no runtime switch.

## Verify

```bash
python3 fixes/qwen3-strict-think-end/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
```

End to end: send a tool-calling request with reasoning enabled many times at the model's sampling
temperature and count calls to `example_function_name` or duplicated calls. With the fix there
are none, provided the model closes `</think>` before its calls.

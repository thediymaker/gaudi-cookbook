# gemma4-buffered-tool-args

Emit `gemma4` tool-call arguments once, complete, with spurious escapes removed.

| | |
| --- | --- |
| Affects | Gemma 4 models served with `--tool-call-parser gemma4` (code-editing and file-writing tools) |
| Releases | v0.26.0, v0.30.0 |
| Patches | `vllm/parser/gemma4.py` |
| Upstream | not filed |
| Test | `test_gemma4_buffered_tool_args.py` (stdlib only) |

## Problem

Gemma 4 writes string arguments between its own `<|"|>` delimiters, and sometimes also
JSON-escapes the quotes and newlines inside them. The `gemma4` parser passes those escapes through,
so an argument such as `\"\"\"Docstring.\"\"\"` reaches the client with literal backslashes.
Every argument JSON goes through `_gemma4_arg_converter(raw_args, partial)`: `partial=True` for
streaming argument diffs, `partial=False` at the end of the call and for non-streaming extraction.

## Symptom

Files written or edited through tool calls contain `\"` and `\n` sequences instead of quotes and
newlines, so generated code does not parse and edits fail to match.

## Who is affected

Anyone serving Gemma 4 with the `gemma4` tool parser whose tools take code or multi-line text
(coding agents, file write and edit tools). Short scalar arguments are rarely affected.

## The fix

- With `partial=True` the converter returns `""`, so no argument diff streams mid-call. The flush
  at the end of the call sends the whole argument JSON as one delta. The tool name and id still
  stream when the call opens.
- With `partial=False` every string value is checked and unescaped only when the escapes are
  evidently spurious: every quote is escaped, the value is JSON-encoded multi-line text, or it is
  Python that parses only after unescaping. Other values are left alone.

The repair is not prefix-stable, which is why arguments are buffered. A call cut off in the middle
of its arguments arrives without arguments instead of with invalid JSON.

## Verify

```bash
python3 fixes/gemma4-buffered-tool-args/apply.py --check   # 0 = applied, 1 = not applied, 2 = source drifted
pytest fixes/gemma4-buffered-tool-args
```

End to end: ask the model, with a `write_file(path, content)` tool, to write a Python module with
a docstring and string literals, streaming and non-streaming. With the fix the `content` argument
parses as Python and contains no stray backslashes.

# llama4-tools-lenient

Source: [`fixes/llama4-tools-lenient/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/llama4-tools-lenient)

A tool-call parser plugin and chat template for Llama 4 that read every call syntax the model
writes and render tool history the way the model expects.

| | |
| --- | --- |
| Affects | Llama 4 Scout and Maverick with tool calling |
| Releases | v0.26.0, v0.30.0 |
| Patches | nothing: `llama4_tools_lenient.py` is a `--tool-parser-plugin`, `llama4-tools.jinja` a `--chat-template` |
| Upstream | not filed yet |
| Test | `test_llama4_tools_lenient.py` (stdlib; the template test needs jinja2) |

## Problem

Llama 4 writes tool calls in several forms: Meta's documented `<function=NAME>{json}</function>`,
Python lists such as `[read(path="a.py")]`, bare calls, calls inside
`<|python_start|> ... <|python_end|>`, and JSON objects. The stock `llama4_pythonic` parser reads
only the Python list form, and only at the start of the reply. The chat template shipped with the
checkpoints re-renders past calls as `param="<raw value>"` (so quotes, newlines and lists break the
model's own history) and JSON-encodes tool results into one escaped line.

## Symptom

Tool calls come back as plain `content`, arguments arrive as strings where the schema wants
numbers, booleans or arrays, and multi-turn agent sessions degrade as the model copies its broken
history. Agents also see the model plan a whole task as one parallel list (read, then an edit of
text it has not read, then a test run), where the blind steps fail.

## Who is affected

Anyone serving Llama 4 with `--enable-auto-tool-choice`.

## The fix

`llama4_tools_lenient.py` registers the `llama4_lenient` parser. It accepts all the forms above,
repairs arguments from the request's tool schemas (no evaluation), streams prose as it arrives and
sends each call whole once it parses. `LLAMA4_TOOL_CALL_POLICY` chooses which calls of a parallel
list are returned:

| Value | Behaviour |
| --- | --- |
| `independent` (default) | only calls that cannot depend on an earlier call in the same turn: a leading run of read-only calls, or of edits to distinct files |
| `reads` | a leading run of read-only calls, else the first call |
| `first` | only the first call (also used when a request sets `parallel_tool_calls: false`) |
| `all` | every call, as upstream |

`LLAMA4_READONLY_TOOLS` and `LLAMA4_EDIT_TOOLS` (comma-separated tool names) override the
built-in tool lists. `llama4-tools.jinja` renders past calls as valid Python literals, passes tool
results verbatim and ends tool turns with `<|eom|>` as Meta's reference encoder does; its options
(`fmt`, `sys_mode`, `defs`, `result`, `eom`, `history`, `seq_rule`) are documented at the top of the
file and set with `--default-chat-template-kwargs`.

`apply.py` writes nothing. It checks that the plugin compiles, the template exists and the
installed vLLM still has the parser API the plugin uses, then prints the serve flags:

```bash
python3 fixes/llama4-tools-lenient/apply.py
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --enable-auto-tool-choice \
  --tool-parser-plugin fixes/llama4-tools-lenient/llama4_tools_lenient.py \
  --tool-call-parser llama4_lenient \
  --chat-template fixes/llama4-tools-lenient/llama4-tools.jinja
```

## Verify

```bash
python3 fixes/llama4-tools-lenient/apply.py   # 0 = plugin usable, 2 = vLLM API drifted
pytest fixes/llama4-tools-lenient
```

End to end: run an agent task that reads, edits and tests a file. Calls should arrive as
`tool_calls` with typed arguments, one dependent step per turn.

# gemma4-bos-tokenizer

Build a tokenizer directory for Gemma 4 instruct checkpoints that adds `<bos>` to raw prompts, so
`/v1/completions` works.

| | |
| --- | --- |
| Affects | Gemma 4 instruct checkpoints (`google/gemma-4-*-it`) served with `/v1/completions` or `/tokenize` |
| Releases | v0.26.0, v0.30.0 (a startup step; it does not patch vLLM) |
| Writes | `OUT_DIR` (a tokenizer directory passed to `--tokenizer`) |
| Upstream | not upstreamable in vLLM: the post-processor comes from the checkpoint's `tokenizer.json` |
| Test | `test_gemma4_bos_tokenizer.py` (stdlib only; the transformers check runs at startup) |

## Problem

Every Gemma sequence starts with `<bos>`. The instruct checkpoints ship a `tokenizer.json` whose
post-processor is `$A` (no `<bos>`) and no `add_bos_token` in `tokenizer_config.json`, because the
chat template writes a literal `<bos>` itself. The base checkpoints use `<bos> $A`. vLLM's
completions path tokenizes raw prompts with `add_special_tokens=True`, which adds nothing here, so
the model sees a prompt without `<bos>`.

## Symptom

`/v1/completions` returns degenerate text for any raw prompt ("LLLLLLLL...", or the prompt
repeated). The same request with a literal `<bos>` at the start of the prompt is correct. Chat
requests are unaffected: the template supplies `<bos>` and chat prompts are tokenized without
special tokens.

## Who is affected

Anyone serving a Gemma 4 instruct checkpoint who sends raw prompts: completions clients, eval
harnesses, perplexity or `prompt_logprobs` checks, and `/tokenize` with default arguments.

## The fix

The step writes `OUT_DIR` as a copy of the model's tokenizer files (symlinks, so vocabulary, chat
template and special tokens are byte-identical) except two files:

- `tokenizer.json`: post-processor `<bos> $A` (and `<bos> $A <bos>:1 $B:1` for pairs), as in the
  base checkpoint;
- `tokenizer_config.json`: `"add_bos_token": true`.

It then checks the result with the installed transformers: a raw prompt gains exactly one leading
`<bos>`, `add_special_tokens=False` is unchanged, vocabulary and chat template are identical, and a
rendered chat prompt tokenizes to the same ids as with the stock tokenizer. Any failure exits 2 and
removes `OUT_DIR`. The directory is rebuilt on every run. If the checkpoint already adds `<bos>`,
its files are linked unchanged.

Run it before the server and pass the directory to `--tokenizer`. Weights, config and the
multimodal processor still load from `--model`.

```bash
python3 fixes/gemma4-bos-tokenizer/apply.py google/gemma-4-31B-it /tmp/gemma4-bos-tokenizer || exit 1
exec vllm serve google/gemma-4-31B-it --tokenizer /tmp/gemma4-bos-tokenizer <other flags>
```

`MODEL` can be a Hugging Face id or a local snapshot directory. With `HF_HUB_OFFLINE=1` the
snapshot must already be cached; otherwise only the small tokenizer and config files are
downloaded. It is not run by `apply_all.py`, because it takes arguments.

## Verify

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "google/gemma-4-31B-it", "prompt": "The capital of France is", "max_tokens": 8}'
pytest fixes/gemma4-bos-tokenizer
```

Without the fix the completion is a run of one repeated token; with it, a normal continuation.
`/tokenize` on the same prompt returns one more id (the `<bos>`) than before.

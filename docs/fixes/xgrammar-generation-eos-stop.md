# xgrammar-generation-eos-stop

Source: [`fixes/xgrammar-generation-eos-stop/`](https://github.com/thediymaker/gaudi-cookbook/tree/main/fixes/xgrammar-generation-eos-stop)

Let a completed grammar end the turn with the model's real end-of-turn tokens, not only the
tokenizer's EOS.

| | |
| --- | --- |
| Affects | structured outputs and forced tool calls (xgrammar backend) on models whose turn ends with a token other than `tokenizer.eos_token_id` |
| Releases | v0.26.0 (v0.30.0 already has the upstream fix; the script reports "not needed") |
| Patches | `vllm/v1/structured_output/backend_xgrammar.py` |
| Upstream | fixed differently by [vllm#49227](https://github.com/vllm-project/vllm/pull/49227) (merged; in the vLLM commit v0.30.0 tracks) |
| Test | `test_xgrammar_generation_eos_stop.py` (stdlib only) |

## Problem

The xgrammar backend builds its `TokenizerInfo` with
`TokenizerInfo.from_huggingface(tokenizer, vocab_size=...)`, so the grammar's only stop token is
`tokenizer.eos_token_id`. Many chat models end a turn with a different token that is listed in
`generation_config.json` under `eos_token_id`. The engine already stops on those, but the grammar
does not know them. After a complete tool call under `tool_choice="required"`, xgrammar allows
only another call or the tokenizer EOS, and masks the real end-of-turn token.

## Symptom

With `tool_choice="required"` (or a named tool), the model keeps emitting tool calls until
`max_tokens`, often dozens of them, and the response ends with `finish_reason: "length"`. GLM-4.7
and the GLM-5 family show it reliably.

## Who is affected

Users of the xgrammar backend (the default choice of `auto`) with models whose
`generation_config.json` lists end tokens that differ from the tokenizer EOS.

## The fix

Pass `stop_token_ids` = tokenizer EOS plus the generation config's `eos_token_id` values (inside
the vocabulary) to xgrammar. A grammar can still stop only where it is complete; the fix only
widens which end token is legal there to the set the engine already treats as end of turn.

vLLM #49227 fixes the same problem per request, by handing each request's stop tokens to the
grammar matcher. The script detects that change and does nothing where it is present.

## Verify

```bash
python3 fixes/xgrammar-generation-eos-stop/apply.py --check   # 0 = applied or not needed, 1 = not applied, 2 = source drifted
pytest fixes/xgrammar-generation-eos-stop
```

End to end: send a chat request with one tool and `tool_choice: "required"`; the response should
contain one call (or a few) and finish with `tool_calls`, not run to `max_tokens`.

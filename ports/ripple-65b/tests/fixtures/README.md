# Fixtures

`config.json`, `generation_config.json` and `tokenizer_config.json` are copied unchanged from
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) at revision
`d58dbcf43344e27494a40cec6727e180450f4a6b`, released by Ricursive under the Apache License 2.0, and so is
`../serving/chat_template.jinja`, which the recipe passes to `--chat-template`. The build gates and CPU tests read
them so that no checkpoint download is needed.

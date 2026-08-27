# Paged prefill acceptance evidence

Source-matched CUDA build: `7d2051be5`, including `c5c197759`.

## Fixed prompts

The commands below create fixed repeated-token prompts. Qwen's prompt is 4096 tokens; Tiel's bounded prompt is 1023 tokens.

```sh
python3 -c "from pathlib import Path; Path('.agent/paged-4096-prompt.txt').write_text(' x' * 4096)"
python3 -c "from pathlib import Path; Path('.agent/paged-1023-prompt.txt').write_text(' x' * 1023)"
```

## Qwen3.8-27B-UD-Q4_K_S

Unified Q8_0 reference, CUDA0, 4096 prompt tokens and 8 generated tokens:

```sh
build-verify-cuda/bin/llama bench -m /models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf -p 4096 -n 8 -r 1 -ngl 99 -dev cuda0 -b 4096 -ub 4096 -ctk q8_0 -ctv q8_0 --no-warmup -o md
```

- pp: 2903.87 tok/s
- tg: 47.25 tok/s

Paged Q8_0, same GPU, context, batch, prompt length, and decode length:

```sh
build-verify-cuda/bin/llama-paged -m /models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -c 8192 -b 4096 -ub 4096 -n 8 -f .agent/paged-4096-prompt.txt -kvp -ctk q8_0 -ctv q8_0 -ngpub 320 -ncpub 320 --kv-paged-watermark 0 -ns 1 -np 1 --timing --perf --temp 0 -s 1234
```

- pp: 54.34 tok/s
- tg: 13.40 tok/s
- completed: 4096 prompt tokens plus 8 generated tokens

## Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S

This hybrid model requires all layers on CUDA0 for paged KV, so both runs use `-ngl 99 -sm none -mg 0`. The bounded case uses 1023 prompt tokens and 8 generated tokens.

Unified Q8_0 reference:

```sh
build-verify-cuda/bin/llama bench -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -p 1023 -n 8 -r 1 -ngl 99 -sm none -mg 0 -dev cuda0 -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 --no-warmup -o md
```

- pp: 1378.04 tok/s
- tg: 129.58 tok/s

Paged Q8_0:

```sh
build-verify-cuda/bin/llama-paged -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -sm none -mg 0 -c 2048 -b 1024 -ub 1024 -n 8 -f .agent/paged-1023-prompt.txt -kvp -ctk q8_0 -ctv q8_0 -ngpub 96 -ncpub 96 --kv-paged-watermark 0 -ns 1 -np 1 --timing --perf --temp 0 -s 1234
```

- pp: 341.76 tok/s
- tg: 61.02 tok/s
- completed: 1023 prompt tokens plus 8 generated tokens

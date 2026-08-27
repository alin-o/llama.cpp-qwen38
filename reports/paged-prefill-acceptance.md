# Paged prefill acceptance evidence

Source-matched CUDA build: `92c7dd195`.

## Fixed prompts

The tracked prompt artifacts make both modes consume identical bytes and tokenizer input:

- `reports/paged-4096-prompt.txt`: SHA-256 `1008707831f454b7333033c82cd4e8c3f3c0105834d34b9365e17d9aeb97e6ae`
- `reports/paged-1023-prompt.txt`: SHA-256 `f05ba13e4bf18bf0964ab81244d82f6a60ac4fd964259a5bd6d7f83613549671`

Each file is deterministic ASCII text: a leading space followed by `x` repeated 4096 or 1023 times. With `-no-cnv`, the recorded completion runs report exactly 4096 or 1023 prompt-eval tokens for the named model tokenizer.

## Qwen3.8-27B-UD-Q4_K_S

Unified Q8_0 reference, CUDA0, the same 4096-token fixed prompt, and 8 generated tokens:

```sh
build-verify-cuda/bin/llama completion -m /models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -c 8192 -b 4096 -ub 4096 -n 8 -f reports/paged-4096-prompt.txt -no-cnv -ctk q8_0 -ctv q8_0 --perf --temp 0 -s 1234
```

- pp: 2913.91 tok/s
- tg: 47.44 tok/s

Paged Q8_0, same GPU, context, batch, prompt length, and decode length:

```sh
build-verify-cuda/bin/llama-paged -m /models/qwen38/Qwen3.8-27B-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -c 8192 -b 4096 -ub 4096 -n 8 -f reports/paged-4096-prompt.txt -kvp -ctk q8_0 -ctv q8_0 -ngpub 320 -ncpub 320 --kv-paged-watermark 0 -ns 1 -np 1 --timing --perf --temp 0 -s 1234
```

- pp: 54.22 tok/s
- tg: 13.23 tok/s
- completed: 4096 prompt tokens plus 8 generated tokens

## Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S

This hybrid model requires all layers on CUDA0 for paged KV, so both runs use `-ngl 99 -sm none -mg 0`. The bounded case uses 1023 prompt tokens and 8 generated tokens.

### 4096-token capacity root cause

The 4096-token paged run was attempted with `-c 8192 -b 4096 -ub 4096 -ngpub 320 -ncpub 320` and the same single-GPU settings. Admission did not deadlock: the context initialization failed while reserving the CUDA prefill compute buffer. The run reported 4074.5 MiB free VRAM, 320 Q8_0 KV blocks at 696320 bytes each, then failed to allocate a 3976.11 MiB CUDA buffer for the prefill graph. The recurrent portion remains on the same CUDA device because hybrid paged KV rejects a CPU/CUDA model split. This is a VRAM compute-buffer limit, not scheduler admission or block accounting. The bounded 1023-token case is therefore used below.

Unified Q8_0 reference, using the same 1023-token fixed prompt and 8 generated tokens:

```sh
build-verify-cuda/bin/llama completion -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -sm none -mg 0 -c 2048 -b 1024 -ub 1024 -n 8 -f reports/paged-1023-prompt.txt -no-cnv -ctk q8_0 -ctv q8_0 --perf --temp 0 -s 1234
```

- pp: 7658.96 tok/s
- tg: 139.27 tok/s

Paged Q8_0:

```sh
build-verify-cuda/bin/llama-paged -m /models/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -dev cuda0 -ngl 99 -sm none -mg 0 -c 2048 -b 1024 -ub 1024 -n 8 -f reports/paged-1023-prompt.txt -kvp -ctk q8_0 -ctv q8_0 -ngpub 96 -ncpub 96 --kv-paged-watermark 0 -ns 1 -np 1 --timing --perf --temp 0 -s 1234
```

- pp: 341.76 tok/s
- tg: 61.02 tok/s
- completed: 1023 prompt tokens plus 8 generated tokens

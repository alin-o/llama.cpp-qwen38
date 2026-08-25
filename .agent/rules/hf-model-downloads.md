---
trigger: model_decision
description: Use when downloading, fetching, or repairing model weights (GGUF) from HuggingFace in this sandbox; covers the hf CLI, the squid egress proxy, the Xet storage failure, and writable paths.
---

# Downloading models from HuggingFace in the sandbox

- Use the `hf` CLI (`/usr/local/bin/hf`, v1.28.0, venv at `/opt/hf-cli/venv`). Form:
  `hf download <repo> <file> --local-dir /models`
  Run with `--dry-run` first to confirm the repo is public, the file exists, and its size before a multi-GB pull.
- All egress goes through a squid proxy (`http://squid:3128`, via HTTP_PROXY/HTTPS_PROXY). HuggingFace hosts (`huggingface.co`, `cdn-lfs.huggingface.co`) return 403 until the user adds a proxy ACL rule. If a download 403s, tell the user the domain needs an ACL allow; do not try to bypass the proxy.
- Xet pitfall: repos served from Xet storage fail with "you need to install the hf_xet package". `hf_xet` is preinstalled but its native lib cannot load here, and the site-packages layer is read-only (pip cannot install/fix it; it errors on `/etc/debian_version`). Work around it by forcing plain HTTP:
  `HF_HUB_DISABLE_XET=1 hf download <repo> <file> --local-dir /models`
- Run long pulls detached: `setsid env HF_HUB_DISABLE_XET=1 /usr/local/bin/hf download ... > repo-dir/pull.log 2>&1 < /dev/null &` — a bare `nohup ... &` inside an agent exec session is killed when the session ends (job dies silently, empty log).
- Always invoke `/usr/local/bin/hf`: the venv's own `bin/hf` script carries a stale absolute shebang into a moved venv path under /home (noexec) and dies with "bad interpreter: Permission denied".
- Writable paths: filesystem root `/` is read-only; `/models` is writable. Download into `/models` via `--local-dir /models`.
- Some `/models` entries are HF/Ollama blob-store symlinks to `../../blobs/<sha256>`. If `blobs/` is not mounted, the symlink is dead. Repair: `rm` the dead symlink, then re-download to the same path as a real file. The sha in the download temp filename matches the symlink target, confirming it is the same artifact.
- Temp files land under `/models/.cache/huggingface/download/` as `*.incomplete`; the finished file appears at the `--local-dir` path. Track progress with `du -sh /models/.cache`.

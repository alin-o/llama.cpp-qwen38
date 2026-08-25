#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path and path.exists() else None


def fmt(value, digits=3):
    return "pending" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--soak", type=Path)
    parser.add_argument("--codex-status", type=Path)
    parser.add_argument("--environment", type=Path, default=Path("docker/results/environment.json"))
    parser.add_argument("--preflight", type=Path, default=Path("docker/results/artifact-preflight.json"))
    parser.add_argument("--checksums", type=Path, default=Path("docker/artifacts.sha256"))
    parser.add_argument("--upstream-lock", type=Path, default=Path("docker/upstream.lock"))
    parser.add_argument("--notes", type=Path, default=Path("docker/results/evaluation-notes.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decision = read_json(args.matrix_dir / "decision.json")
    quality = read_json(args.quality)
    soak = read_json(args.soak)
    environment = read_json(args.environment)
    preflight = read_json(args.preflight)
    checksums = args.checksums.read_text().strip() if args.checksums.exists() else "unavailable"
    upstream_lock = args.upstream_lock.read_text().strip() if args.upstream_lock.exists() else "unavailable"
    notes = args.notes.read_text().strip() if args.notes.exists() else ""
    codex_pass = args.codex_status is not None and args.codex_status.exists() and args.codex_status.read_text().strip() == "PASS"
    candidate = decision.get("preliminary_candidate") if decision else None
    candidate_key = f"upstream-{candidate}" if candidate else None
    candidate_result = next((item for item in decision.get("profiles", []) if item.get("profile") == candidate), None) if decision else None
    quality_pass = bool(quality and candidate_key in quality.get("profiles", {}) and quality["profiles"][candidate_key]["within_one_percent"])
    long_pass = bool(candidate_result and candidate_result["gates"].get("long_context_11_of_12"))
    soak_pass = bool(soak and soak.get("passed"))
    short_pass = bool(candidate_result and candidate_result.get("completed_gates_pass"))
    final_pass = short_pass and quality_pass and long_pass and soak_pass and codex_pass

    lines = [
        "# Qwen3.8 27B upstream migration report",
        "",
        f"- Decision: {'PASS - switch to upstream' if final_pass else 'PENDING/FAIL - keep Qwen3.6 production'}",
        f"- Production cache profile: {candidate if final_pass else 'none'}",
        f"- Preliminary short-run profile: {candidate or 'none'}",
        f"- Short gates: {'preliminary pass' if short_pass else 'fail or pending'}",
        f"- WikiText PPL gate: {'pass' if quality_pass else 'fail or pending'}",
        f"- Long-context gate: {'pass' if long_pass else 'fail or pending'}",
        f"- 24-hour soak: {'pass' if soak_pass else 'fail or pending'}",
        f"- Codex fixture: {'pass' if codex_pass else 'fail or pending'}",
        "",
        "## Pinned inputs and environment",
        "",
        "```text",
        upstream_lock,
        "",
        checksums,
        "```",
        "",
        "## Evaluation notes",
        "",
        notes or "No additional notes.",
        "",
        "## Recorded environment",
        "",
        "```json",
        json.dumps(environment, indent=2, sort_keys=True),
        "```",
        "",
        "## Profile results",
        "",
        "| Profile | Completed gates | Free VRAM MiB | Base decode/fork | MTP decode/fork | MTP gain |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in decision.get("profiles", []) if decision else []:
        ratios = item.get("ratios", {})
        pending = item.get("pending_gates") or []
        gate_status = "fail" if not item.get("completed_gates_pass") else f"preliminary pass; {len(pending)} pending"
        lines.append(
            f"| {item['profile']} | {gate_status} | {fmt(item.get('minimum_free_mib'), 0)} | {fmt(ratios.get('base_decode_vs_fork'))} | {fmt(ratios.get('mtp_decode_vs_fork'))} | {fmt(ratios.get('mtp_gain'))} |"
        )
    if quality:
        lines.extend([
            "",
            "## WikiText-2 perplexity",
            "",
            f"- Fork TurboQuant3: {quality.get('fork_turbo3_ppl')}",
            "",
            "```json",
            json.dumps(quality.get("profiles", {}), indent=2, sort_keys=True),
            "```",
        ])
    lines.extend(["", "## Artifact preflight", "", "```json", json.dumps(preflight, indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Gate details", "", "```json", json.dumps(decision, indent=2, sort_keys=True), "```", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)
    return 0 if final_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

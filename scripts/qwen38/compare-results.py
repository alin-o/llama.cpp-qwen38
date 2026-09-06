#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def load_json(path):
    if not path.exists():
        return {"errors": [f"missing result: {path.name}"]}
    return json.loads(path.read_text())


def speed(report, key):
    value = (report.get("summary") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def long_speed(report, key):
    values = []
    for row in report.get("long_context") or []:
        if row.get("target_tokens") != 100000:
            continue
        value = (row.get("timings") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.median(values) if values else None


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def exact_parity(base, speculative):
    base_rows = base.get("prompts") or []
    spec_rows = speculative.get("prompts") or []
    if len(base_rows) != 20 or len(spec_rows) != 20:
        return False, []
    mismatches = []
    for base_row, spec_row in zip(base_rows, spec_rows):
        if base_row.get("content") != spec_row.get("content"):
            mismatches.append(base_row.get("index"))
    return not mismatches, mismatches


def minimum_free_mib(path):
    if not path.exists():
        return None
    values = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 6:
                continue
            try:
                values.append(float(row[5].strip()))
            except ValueError:
                continue
    return min(values) if values else None


def api_ok(report):
    return not report.get("errors") and all(key in report for key in (
        "non_streaming",
        "streaming",
        "reasoning",
        "tool_call",
        "vision",
        "slot_round_trip",
    ))


def long_context_gate(report):
    rows = report.get("long_context")
    if rows is None:
        return None, None
    passed = sum(bool(row.get("passed")) for row in rows)
    capacity = report.get("capacity_generation") or {}
    capacity_ok = bool(capacity.get("passed"))
    return passed >= 11 and capacity_ok, {
        "passed": passed,
        "total": len(rows),
        "capacity_generation": capacity,
    }


def evaluate(results_dir, profile, fork_base, fork_mtp):
    upstream_base = load_json(results_dir / f"upstream-{profile}-base.json")
    upstream_mtp = load_json(results_dir / f"upstream-{profile}-mtp.json")
    parity_ok, mismatches = exact_parity(upstream_base, upstream_mtp)
    min_free = minimum_free_mib(results_dir / f"upstream-{profile}-mtp.gpu.csv")
    long_ok, long_summary = long_context_gate(upstream_mtp)

    # Tiny chat prompts are dominated by template-prefix caching and fixed
    # graph overhead. Prompt-processing parity is therefore measured only on
    # the required fresh 100k workload. Decode uses the same 100k cases when
    # available and falls back to the 20 deterministic short prompts while
    # the long-context gate is still pending.
    base_prompt_ratio = ratio(
        long_speed(upstream_base, "prompt_per_second"),
        long_speed(fork_base, "prompt_per_second"),
    )
    mtp_prompt_ratio = ratio(
        long_speed(upstream_mtp, "prompt_per_second"),
        long_speed(fork_mtp, "prompt_per_second"),
    )
    base_decode_ratio = ratio(
        long_speed(upstream_base, "predicted_per_second") or speed(upstream_base, "median_decode_t_s"),
        long_speed(fork_base, "predicted_per_second") or speed(fork_base, "median_decode_t_s"),
    )
    mtp_decode_ratio = ratio(
        long_speed(upstream_mtp, "predicted_per_second") or speed(upstream_mtp, "median_decode_t_s"),
        long_speed(fork_mtp, "predicted_per_second") or speed(fork_mtp, "median_decode_t_s"),
    )
    mtp_gain = ratio(speed(upstream_mtp, "median_decode_t_s"), speed(upstream_base, "median_decode_t_s"))

    gates = {
        "api_base": api_ok(upstream_base),
        "api_mtp": api_ok(upstream_mtp),
        "greedy_mtp_parity": parity_ok,
        "one_gib_vram_headroom": min_free is not None and min_free >= 1024,
        "base_prompt_within_10_percent": None if base_prompt_ratio is None else base_prompt_ratio >= 0.90,
        "base_decode_within_10_percent": base_decode_ratio is not None and base_decode_ratio >= 0.90,
        "mtp_prompt_within_10_percent": None if mtp_prompt_ratio is None else mtp_prompt_ratio >= 0.90,
        "mtp_decode_within_10_percent": mtp_decode_ratio is not None and mtp_decode_ratio >= 0.90,
        "mtp_gain_at_least_3_percent": mtp_gain is not None and mtp_gain >= 1.03,
        "long_context_11_of_12": long_ok,
    }
    completed = {key: value for key, value in gates.items() if value is not None}
    return {
        "profile": profile,
        "gates": gates,
        "completed_gates_pass": all(completed.values()),
        "pending_gates": [key for key, value in gates.items() if value is None],
        "minimum_free_mib": min_free,
        "ratios": {
            "base_prompt_vs_fork": base_prompt_ratio,
            "base_decode_vs_fork": base_decode_ratio,
            "mtp_prompt_vs_fork": mtp_prompt_ratio,
            "mtp_decode_vs_fork": mtp_decode_ratio,
            "mtp_gain": mtp_gain,
        },
        "parity_mismatches": mismatches,
        "long_context": long_summary,
        "errors": {
            "base": upstream_base.get("errors") or [],
            "mtp": upstream_mtp.get("errors") or [],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fork_base = load_json(args.results_dir / "fork-turbo3-base.json")
    fork_mtp = load_json(args.results_dir / "fork-turbo3-nextn.json")
    profiles = [
        evaluate(args.results_dir, "q8q5", fork_base, fork_mtp),
        evaluate(args.results_dir, "iq4nl", fork_base, fork_mtp),
    ]
    candidate = next((item["profile"] for item in profiles if item["completed_gates_pass"]), None)
    report = {
        "preliminary_candidate": candidate,
        "profiles": profiles,
        "global_pending_gates": ["wikitext_perplexity", "24_hour_soak", "codex_coding_session"],
        "decision": "pending" if candidate else "no_preliminary_candidate",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

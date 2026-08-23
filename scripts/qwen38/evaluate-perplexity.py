#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PATTERNS = [
    re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]


def parse(path):
    text = path.read_text(errors="replace")
    values = []
    for pattern in PATTERNS:
        values = [float(value) for value in pattern.findall(text)]
        if values:
            break
    if not values:
        raise RuntimeError(f"no perplexity value found in {path}")
    return values[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fork = parse(args.results_dir / "fork-turbo3.log")
    profiles = {}
    for profile in ("upstream-q8q5", "upstream-iq4nl"):
        ppl = parse(args.results_dir / f"{profile}.log")
        delta = ppl / fork - 1.0
        profiles[profile] = {
            "ppl": ppl,
            "relative_delta": delta,
            "within_one_percent": delta <= 0.01,
        }
    report = {"fork_turbo3_ppl": fork, "profiles": profiles}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if any(item["within_one_percent"] for item in profiles.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())


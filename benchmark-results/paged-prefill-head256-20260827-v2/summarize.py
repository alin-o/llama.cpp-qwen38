#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


OUT = Path(__file__).resolve().parent


def summarize_paged():
    groups = defaultdict(list)
    with (OUT / "paged-raw.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            key = tuple(row[name] for name in (
                "model", "prompt_tokens", "block_size", "ubatch_size", "requests"
            ))
            groups[key].append(row)

    with (OUT / "paged-medians.csv").open("w", newline="") as target:
        fieldnames = [
            "model", "prompt_tokens", "block_size", "ubatch_size", "requests",
            "measured_repetitions", "median_pp_tok_s", "median_tg_tok_s", "status", "raw_log",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in groups.items():
            measured = rows[1:]
            source_status = rows[0]["status"]
            pp = [float(row["pp_tok_s"]) for row in measured if row["pp_tok_s"]]
            tg = [float(row["tg_tok_s"]) for row in measured if row["tg_tok_s"]]
            if source_status.startswith("exit_"):
                status = "capacity_failure"
            elif len(pp) == 3 and len(tg) == 3 and max(pp) == 0 and max(tg) == 0:
                status = "scheduler_rejection"
            elif len(pp) == 3 and len(tg) == 3 and min(pp) > 0 and min(tg) > 0:
                status = "executed"
            else:
                status = "incomplete"
            writer.writerow({
                **dict(zip(fieldnames[:5], key)),
                "measured_repetitions": 3 if len(pp) == 3 and len(tg) == 3 else 0,
                "median_pp_tok_s": f"{median(pp):.2f}" if pp and max(pp) > 0 else "",
                "median_tg_tok_s": f"{median(tg):.2f}" if tg and max(tg) > 0 else "",
                "status": status,
                "raw_log": rows[0]["raw_log"],
            })


def summarize_unified():
    raw_rows = []
    median_rows = []
    for path in sorted((OUT / "raw" / "unified").glob("*.json")):
        model, ubatch_part = path.stem.split("-")
        ubatch = int(ubatch_part.removeprefix("ub"))
        with path.open() as source:
            results = json.load(source)
        tg_samples = None
        for result in results:
            phase = "pp" if result["n_prompt"] else "tg"
            tokens = result["n_prompt"] or result["n_gen"]
            samples = result["samples_ts"]
            for repetition, value in enumerate(samples, start=1):
                raw_rows.append({
                    "model": model,
                    "prompt_tokens": result["n_prompt"],
                    "ubatch_size": ubatch,
                    "phase": phase,
                    "tokens": tokens,
                    "repetition": repetition,
                    "tok_s": value,
                    "raw_log": f"raw/unified/{path.name}",
                })
            if phase == "tg":
                tg_samples = samples
            else:
                median_rows.append({
                    "model": model,
                    "prompt_tokens": result["n_prompt"],
                    "ubatch_size": ubatch,
                    "median_pp_tok_s": f"{median(samples):.2f}",
                    "median_tg_tok_s": "",
                    "measured_repetitions": len(samples),
                    "raw_log": f"raw/unified/{path.name}",
                })
        if tg_samples is not None:
            for row in median_rows:
                if row["model"] == model and row["ubatch_size"] == ubatch:
                    row["median_tg_tok_s"] = f"{median(tg_samples):.2f}"

    with (OUT / "unified-raw.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    with (OUT / "unified-medians.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=median_rows[0].keys())
        writer.writeheader()
        writer.writerows(median_rows)


summarize_paged()
summarize_unified()


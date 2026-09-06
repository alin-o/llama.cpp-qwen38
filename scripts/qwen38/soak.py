#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from acceptance import (
    PROMPTS,
    chat,
    check_slot_round_trip,
    check_streaming,
    check_tool_call,
    check_vision,
    content_of,
    wait_ready,
)


def acceptance_ratio(rows):
    drafted = sum(int((row.get("timings") or {}).get("draft_n") or 0) for row in rows)
    accepted = sum(int((row.get("timings") or {}).get("draft_n_accepted") or 0) for row in rows)
    return accepted / drafted if drafted else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8882")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--minimum-requests", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_ready(base_url, 1800)
    started = time.time()
    deadline = started + args.hours * 3600
    rows = []
    errors = []
    request_index = 0
    while time.time() < deadline or request_index < args.minimum_requests:
        prompt = PROMPTS[request_index % len(PROMPTS)]
        effort = ("low", "medium", "high", "xhigh")[request_index % 4]
        try:
            before = time.perf_counter()
            row = {
                "index": request_index,
                "effort": effort,
            }
            if request_index % 20 == 19:
                row["kind"] = "vision"
                row["result"] = check_vision(base_url)
            elif request_index % 15 == 14:
                row["kind"] = "tool_call"
                row["result"] = check_tool_call(base_url)
            elif request_index % 10 == 9:
                row["kind"] = "streaming"
                row["result"] = check_streaming(base_url)
            else:
                row["kind"] = "chat"
                response = chat(
                    base_url,
                    [{"role": "user", "content": prompt}],
                    max_tokens=512,
                    reasoning_effort=effort,
                    cache_prompt=False,
                )
                message = response["choices"][0]["message"]
                visible = content_of(response)
                reasoning = message.get("reasoning_content") or ""
                if not visible and not reasoning:
                    raise RuntimeError("chat response has neither content nor reasoning_content")
                row.update({
                    "content_length": len(visible),
                    "reasoning_length": len(reasoning),
                    "timings": response.get("timings") or {},
                })
            row["elapsed_s"] = time.perf_counter() - before
            rows.append(row)
            if request_index and request_index % 25 == 0:
                check_slot_round_trip(base_url)
        except Exception as exc:
            errors.append({"index": request_index, "error": str(exc)})
        request_index += 1
        if time.time() < deadline:
            time.sleep(max(1.0, args.hours * 3600 / max(args.minimum_requests, 1) - 1.0))

    midpoint = len(rows) // 2
    first_acceptance = acceptance_ratio(rows[:midpoint])
    second_acceptance = acceptance_ratio(rows[midpoint:])
    acceptance_not_declining = (
        first_acceptance is not None
        and second_acceptance is not None
        and second_acceptance >= first_acceptance - 0.05
    )
    duration_ok = time.time() - started >= args.hours * 3600
    report = {
        "started": started,
        "finished": time.time(),
        "requested_hours": args.hours,
        "minimum_requests": args.minimum_requests,
        "completed_requests": len(rows),
        "errors": errors,
        "rows": rows,
        "mtp_acceptance": {
            "first_half": first_acceptance,
            "second_half": second_acceptance,
            "not_declining_by_more_than_5_points": acceptance_not_declining,
        },
        "passed": len(rows) >= args.minimum_requests and not errors and duration_ok and acceptance_not_declining,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("completed_requests", "errors", "passed")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

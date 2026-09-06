#!/usr/bin/env python3
import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = [
    "Return only the number obtained by adding 17 and 25.",
    "Write a Python function that returns the larger of two integers.",
    "Explain in one sentence why HTTPS uses certificates.",
    "Return the lowercase form of TURBOQUANT.",
    "What is the next prime after 97? Return only the number.",
    "Write a valid JSON object with keys name and enabled.",
    "Name the capital of Romania. Return only the city.",
    "What does a mutex protect? Answer in one sentence.",
    "Convert 5 GiB to bytes. Return only the integer.",
    "Write a Bash conditional that tests whether file.txt exists.",
    "Give the hexadecimal representation of decimal 255.",
    "What HTTP status means Not Found? Return only the number.",
    "Sort these words alphabetically: pear apple orange.",
    "Write a C++ statement that increments variable count.",
    "Return only the result of 12 * 13.",
    "State one difference between RAM and disk in one sentence.",
    "Write a regular expression that matches one or more digits.",
    "What port does HTTPS normally use? Return only the number.",
    "Return a JSON array containing 1, 2, and 3.",
    "Write one sentence describing speculative decoding.",
]

PNG_1X1 = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZlZsAAAAASUVORK5CYII="
)).decode()


def http_json(url, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc


def http_text(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode(errors="replace")


def chat(base_url, messages, max_tokens=128, **extra):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "stream": False,
    }
    payload.update(extra)
    return http_json(base_url + "/v1/chat/completions", payload)


def content_of(response):
    return response["choices"][0]["message"].get("content") or ""


def wait_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            health = http_json(base_url + "/health", timeout=10)
            if health.get("status") == "ok":
                return health
        except Exception as exc:
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"server did not become ready: {last_error}")


def check_streaming(base_url):
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(base_url + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
    chunks = 0
    done = False
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            value = line[6:]
            if value == "[DONE]":
                done = True
                break
            json.loads(value)
            chunks += 1
    if chunks == 0 or not done:
        raise RuntimeError("streaming response was incomplete")
    return {"chunks": chunks, "done": done}


def check_tool_call(base_url):
    response = chat(
        base_url,
        [{"role": "user", "content": "Use get_weather for Bucharest. Do not answer from memory."}],
        max_tokens=128,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        tool_choice="required",
    )
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0].get("function", {}).get("name") != "get_weather":
        raise RuntimeError(f"structured tool call missing: {calls!r}")
    arguments = calls[0]["function"].get("arguments", "{}")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if "bucharest" not in str(arguments.get("city", "")).lower():
        raise RuntimeError(f"tool arguments are incorrect: {arguments!r}")
    return calls[0]


def check_reasoning(base_url):
    results = {}
    for effort in ("low", "medium", "high", "xhigh"):
        response = chat(
            base_url,
            [{"role": "user", "content": "Reply with OK."}],
            max_tokens=16,
            reasoning_effort=effort,
        )
        results[effort] = response["choices"][0]["finish_reason"]
    preserved = chat(
        base_url,
        [
            {"role": "user", "content": "Remember the number 731."},
            {"role": "assistant", "reasoning_content": "The user asked me to remember 731.", "content": "Understood."},
            {"role": "user", "content": "Reply only with the remembered number."},
        ],
        max_tokens=512,
        chat_template_kwargs={"preserve_reasoning": True},
    )
    results["preserve_reasoning"] = content_of(preserved)
    if "731" not in results["preserve_reasoning"]:
        raise RuntimeError(
            "multi-turn reasoning preservation lost the remembered value: "
            + repr(results["preserve_reasoning"])
        )
    return results


def check_vision(base_url):
    response = chat(
        base_url,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG_1X1}},
                {"type": "text", "text": "Reply with OK if the image was processed."},
            ],
        }],
        # Qwen3.8 emits a short reasoning trace before its visible answer. 32
        # tokens can end exactly after reasoning and falsely report an empty
        # multimodal response even though the projector ran successfully.
        max_tokens=128,
    )
    if not content_of(response).strip():
        raise RuntimeError("vision response is empty")
    return content_of(response)


def check_slot_round_trip(base_url):
    chat(base_url, [{"role": "user", "content": "Reply with SLOT-OK."}], max_tokens=32)
    saved = http_json(base_url + "/slots/0?action=save", {"filename": "qwen38-acceptance.bin"})
    erased = http_json(base_url + "/slots/0?action=erase", {})
    restored = http_json(base_url + "/slots/0?action=restore", {"filename": "qwen38-acceptance.bin"})
    if saved.get("n_saved", 0) <= 0 or restored.get("n_restored", 0) <= 0:
        raise RuntimeError(f"slot save/restore did not preserve tokens: {saved!r} {restored!r}")
    return {"save": saved, "erase": erased, "restore": restored}


def capture_prompts(base_url, max_tokens):
    rows = []
    for index, prompt in enumerate(PROMPTS):
        started = time.perf_counter()
        # Cached and fresh prefixes can use different batch shapes and are not
        # guaranteed bit-identical. Disable reuse so this exact-output gate
        # isolates speculative decoding from prompt-cache nondeterminism.
        response = chat(
            base_url,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            cache_prompt=False,
        )
        elapsed = time.perf_counter() - started
        timings = response.get("timings") or {}
        rows.append({
            "index": index,
            "prompt": prompt,
            "content": content_of(response).strip(),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "elapsed_s": elapsed,
            "prompt_t_s": timings.get("prompt_per_second"),
            "decode_t_s": timings.get("predicted_per_second"),
            "draft_n": timings.get("draft_n", 0),
            "draft_n_accepted": timings.get("draft_n_accepted", 0),
        })
    return rows


def token_count(base_url, content):
    response = http_json(base_url + "/tokenize", {"content": content, "add_special": False})
    return len(response["tokens"])


def long_context_case(base_url, target_tokens, position, case_index):
    key = f"NEEDLE-{target_tokens}-{position}-{case_index}-QWEN38"
    filler = "This archival line contains routine information and no secret key.\n"
    repeats = max(1, target_tokens // 12)
    for _ in range(3):
        sample = filler * repeats
        count = token_count(base_url, sample)
        repeats = max(1, int(repeats * (target_tokens - 128) / max(count, 1)))
    lines = [filler] * repeats
    insertion = int(len(lines) * position)
    lines.insert(insertion, f"The exact retrieval key is {key}.\n")
    context = "".join(lines)
    count = token_count(base_url, context)
    response = chat(
        base_url,
        [{"role": "user", "content": context + "\nReturn only the exact retrieval key."}],
        max_tokens=512,
        cache_prompt=False,
        reasoning_effort="low",
    )
    answer = content_of(response).strip()
    return {
        "target_tokens": target_tokens,
        "actual_tokens": count,
        "position": position,
        "key": key,
        "answer": answer,
        "passed": key in answer,
        "timings": response.get("timings") or {},
    }


def capacity_generation_case(base_url):
    filler = "This is deterministic capacity-test text for the Qwen3.8 migration.\n"
    repeats = 100000 // 12
    for _ in range(4):
        context = filler * repeats
        count = token_count(base_url, context)
        repeats = max(1, int(repeats * 100000 / max(count, 1)))
    context = filler * repeats
    response = chat(
        base_url,
        [{
            "role": "user",
            "content": context + "\nContinue with a numbered technical checklist.",
        }],
        max_tokens=2048,
        cache_prompt=False,
        ignore_eos=True,
        reasoning_effort="low",
    )
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "requested_prompt_tokens": 100000,
        "prompt_tokens": prompt_tokens,
        "requested_completion_tokens": 2048,
        "completion_tokens": completion_tokens,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "timings": response.get("timings") or {},
        "passed": prompt_tokens >= 100000 and completion_tokens >= 2048,
    }


def run_long_context(base_url):
    rows = []
    for target in (8192, 32768, 65536, 100000):
        for case_index, position in enumerate((0.25, 0.50, 0.75)):
            rows.append(long_context_case(base_url, target, position, case_index))
    return rows


def summarize(rows):
    prompt_speeds = [row["prompt_t_s"] for row in rows if isinstance(row["prompt_t_s"], (int, float))]
    decode_speeds = [row["decode_t_s"] for row in rows if isinstance(row["decode_t_s"], (int, float))]
    drafted = sum(int(row["draft_n"] or 0) for row in rows)
    accepted = sum(int(row["draft_n_accepted"] or 0) for row in rows)
    return {
        "median_prompt_t_s": statistics.median(prompt_speeds) if prompt_speeds else None,
        "median_decode_t_s": statistics.median(decode_speeds) if decode_speeds else None,
        "draft_n": drafted,
        "draft_n_accepted": accepted,
        "acceptance": accepted / drafted if drafted else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8882")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    report = {"profile": args.profile, "base_url": base_url, "started": time.time(), "errors": []}
    try:
        report["health"] = wait_ready(base_url, args.ready_timeout)
        report["non_streaming"] = content_of(chat(base_url, [{"role": "user", "content": "Reply with OK."}], max_tokens=16))
        report["streaming"] = check_streaming(base_url)
        report["reasoning"] = check_reasoning(base_url)
        report["tool_call"] = check_tool_call(base_url)
        report["vision"] = check_vision(base_url)
        try:
            report["slot_round_trip"] = check_slot_round_trip(base_url)
        except RuntimeError as exc:
            # The historical fork cannot serialize a slot while its custom
            # multimodal implementation is enabled. Keep that limitation in
            # the reference report without losing its performance baseline;
            # upstream profiles must still pass slot save/restore.
            if args.profile.startswith("fork-") and "not supported by multimodal" in str(exc).lower():
                report["slot_round_trip"] = {"supported": False, "error": str(exc)}
            else:
                raise
        report["prompts"] = capture_prompts(base_url, args.max_tokens)
        report["summary"] = summarize(report["prompts"])
        report["metrics"] = http_text(base_url + "/metrics")
        if args.long_context:
            report["long_context"] = run_long_context(base_url)
            report["capacity_generation"] = capacity_generation_case(base_url)
    except Exception as exc:
        report["errors"].append(str(exc))
    report["finished"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))
    if report["errors"]:
        print("error: " + "; ".join(report["errors"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

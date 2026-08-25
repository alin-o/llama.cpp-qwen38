#!/usr/bin/env python3
"""Benchmark a llama.cpp proxy with/without MTP.
Hits the proxy's /v1/chat/completions, retries past errors (GPU contention),
and reports median throughput + MTP draft acceptance over N successful runs.
Usage: bench_client.py <base_url> <prompt_file> --out <toks> --temp t --runs N
"""
import sys, json, time, argparse, statistics, urllib.request, urllib.error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("prompt_file")
    ap.add_argument("--out", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--maxtries", type=int, default=30)
    args = ap.parse_args()

    prompt = open(args.prompt_file, encoding="utf-8").read()
    body = json.dumps({
        "model": "tielcoder",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.out, "temperature": args.temp,
        "top_p": args.top_p, "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"})

    rows = []
    tries = 0
    while len(rows) < args.runs and tries < args.maxtries:
        tries += 1
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.load(r)
        except Exception as e:
            print(f"  try {tries}: error ({e.__class__.__name__}); retry", file=sys.stderr)
            time.sleep(3)
            continue
        t = d.get("timings", {})
        if not t.get("predicted_per_second"):
            print(f"  try {tries}: odd timings {t}", file=sys.stderr)
            continue
        u = d.get("usage", {})
        dn, dna = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        rows.append({
            "gen_tps": t.get("predicted_per_second"),
            "prompt_tps": t.get("prompt_per_second"),
            "ttft": t.get("prompt_ms", 0) / 1000.0,
            "draft_n": dn, "dna": dna,
            "acc": 100.0 * dna / max(1, dn),
        })
        print(f"  ok gen={t.get('predicted_per_second'):.0f}tok/s "
              f"prompt={t.get('prompt_per_second'):.0f}tok/s "
              f"ttft={t.get('prompt_ms',0)/1000:.2f}s MTP={dna}/{dn}={100.0*dna/max(1,dn):.0f}%")

    if not rows:
        print("no successful runs", file=sys.stderr); sys.exit(1)
    g = [r["gen_tps"] for r in rows]
    p = [r["prompt_tps"] for r in rows]
    tt = [r["ttft"] for r in rows]
    ac = [r["acc"] for r in rows]
    print("=" * 64)
    print(f"runs={len(rows)}  gen tok/s   med={statistics.median(g):.0f} "
          f"min={min(g):.0f} max={max(g):.0f}")
    print(f"prompt tok/s  med={statistics.median(p):.0f} "
          f"min={min(p):.0f} max={max(p):.0f}")
    print(f"TTFT          med={statistics.median(tt):.2f}s "
          f"min={min(tt):.2f} max={max(tt):.2f}")
    print(f"MTP accept    med={statistics.median(ac):.0f}% "
          f"min={min(ac):.0f} max={max(ac):.0f}%")

if __name__ == "__main__":
    main()

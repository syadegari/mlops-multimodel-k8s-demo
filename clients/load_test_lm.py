#!/usr/bin/env python3
"""
Load-test the LM service by sending many /generate requests concurrently.

Usage (from repo root, with port-forward or service URL):

    python clients/load_test_lm.py \
        --url http://127.0.0.1:8100/generate \
        --duration-seconds 120 \
        --concurrency 20 \
        --max-new-tokens 40

Optionally, you can provide a file with one prompt per line:

    python clients/load_test_lm.py \
        --url http://127.0.0.1:8100/generate \
        --prompts-file data/lm_prompts.txt
"""

import argparse
import random
import threading
import time
from typing import List

import requests


DEFAULT_PROMPTS = [
    "Explain what a satellite image can tell us about vegetation.",
    "Describe possible land use types visible from space.",
    "What patterns in a time series of images might indicate flooding?",
    "Write a short description of clouds over the ocean.",
    "Summarize what an analyst might look for in EO data.",
]


class LoadStats:
    def __init__(self) -> None:
        self.total_requests = 0
        self.total_success = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self.max_latency = 0.0
        self._lock = threading.Lock()

    def record(self, success: bool, latency: float) -> None:
        with self._lock:
            self.total_requests += 1
            if success:
                self.total_success += 1
            else:
                self.total_errors += 1
            self.total_latency += latency
            self.max_latency = max(self.max_latency, latency)

    def snapshot(self):
        with self._lock:
            if self.total_requests > 0:
                avg = self.total_latency / self.total_requests
            else:
                avg = 0.0
            return {
                "total_requests": self.total_requests,
                "total_success": self.total_success,
                "total_errors": self.total_errors,
                "avg_latency": avg,
                "max_latency": self.max_latency,
            }


def load_prompts_from_file(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def worker(
    worker_id: int,
    url: str,
    prompts: List[str],
    max_new_tokens: int,
    end_time: float,
    stats: LoadStats,
    timeout: float = 10.0,
) -> None:
    session = requests.Session()
    rng = random.Random(worker_id + int(time.time()))
    n = len(prompts)

    while time.perf_counter() < end_time:
        prompt = prompts[rng.randrange(n)]
        payload = {"prompt": prompt, "max_new_tokens": max_new_tokens}
        start = time.perf_counter()
        success = False
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            success = resp.status_code == 200
        except Exception:
            success = False
        latency = time.perf_counter() - start
        stats.record(success=success, latency=latency)


def parse_args():
    p = argparse.ArgumentParser(description="Load-test the LM /generate endpoint.")
    p.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8100/generate",
        help="URL of the LM /generate endpoint.",
    )
    p.add_argument(
        "--duration-seconds",
        type=int,
        default=120,
        help="How long to run the load test (in seconds).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent worker threads.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="max_new_tokens to send in each request.",
    )
    p.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Optional path to a text file with one prompt per line.",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="How often (in seconds) to print intermediate stats.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[load_test_lm] Target URL: {args.url}")
    print(
        f"[load_test_lm] duration={args.duration_seconds}s, "
        f"concurrency={args.concurrency}, max_new_tokens={args.max_new_tokens}"
    )

    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"[load_test_lm] Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"[load_test_lm] Using {len(prompts)} built-in prompts")

    if not prompts:
        raise SystemExit("[load_test_lm] No prompts available; aborting.")

    stats = LoadStats()
    end_time = time.perf_counter() + args.duration_seconds

    threads = []
    for wid in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(wid, args.url, prompts, args.max_new_tokens, end_time, stats),
            daemon=True,
        )
        t.start()
        threads.append(t)

    start_time = time.perf_counter()
    next_log = start_time + args.log_interval

    try:
        while True:
            now = time.perf_counter()
            if now >= end_time:
                break
            if now >= next_log:
                elapsed = now - start_time
                snap = stats.snapshot()
                rps = snap["total_requests"] / elapsed if elapsed > 0 else 0.0
                print(
                    f"[load_test_lm] t={elapsed:5.1f}s | "
                    f"req={snap['total_requests']} | "
                    f"succ={snap['total_success']} | "
                    f"err={snap['total_errors']} | "
                    f"rps={rps:6.1f} | "
                    f"avg_lat={snap['avg_latency']*1000:6.1f} ms | "
                    f"max_lat={snap['max_latency']*1000:6.1f} ms"
                )
                next_log += args.log_interval
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[load_test_lm] Interrupted by user; waiting for workers to finish...")

    for t in threads:
        t.join(timeout=1.0)

    elapsed = time.perf_counter() - start_time
    snap = stats.snapshot()
    rps = snap["total_requests"] / elapsed if elapsed > 0 else 0.0

    print("\n[load_test_lm] FINAL STATS")
    print(f"  elapsed:       {elapsed:.1f} s")
    print(f"  requests:      {snap['total_requests']}")
    print(f"  successes:     {snap['total_success']}")
    print(f"  errors:        {snap['total_errors']}")
    print(f"  RPS:           {rps:.1f}")
    print(f"  avg latency:   {snap['avg_latency']*1000:.1f} ms")
    print(f"  max latency:   {snap['max_latency']*1000:.1f} ms")


if __name__ == "__main__":
    main()

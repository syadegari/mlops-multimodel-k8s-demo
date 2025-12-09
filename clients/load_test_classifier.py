#!/usr/bin/env python3
"""
Load-test the classifier service by sending many /predict requests concurrently.

Usage (from repo root, with port-forward or service URL):

    # Local uvicorn or port-forwarded service
    python clients/load_test_classifier.py \
        --url http://127.0.0.1:8000/predict \
        --duration-seconds 60 \
        --concurrency 10 \
        --dataset-size 1000

This will:
  - load MNIST once (cached on disk after first run),
  - take the first N samples as a pool of requests,
  - start N worker threads that continuously send requests until duration is over,
  - print live stats (RPS, errors, avg latency).
"""

import argparse
import random
import threading
import time
from typing import List, Tuple

import numpy as np
import requests
from sklearn.datasets import fetch_openml


def load_mnist_subset(dataset_size: int) -> Tuple[List[List[float]], List[int]]:
    """
    Load MNIST, scale to [0,1], and return a subset of feature vectors + labels.

    Returns:
        features_list: list of feature lists suitable for JSON payloads
        labels:        list of int labels (for sanity checks; not used in traffic)
    """
    print("[load_test_classifier] Fetching MNIST from OpenML (cached after first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(int)

    n = min(dataset_size, X.shape[0])
    X = X[:n]
    y = y[:n]

    print(f"[load_test_classifier] Using subset: X shape = {X.shape}, y shape = {y.shape}")

    features_list = [row.tolist() for row in X]
    labels = [int(label) for label in y]
    return features_list, labels


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
            if latency > self.max_latency:
                self.max_latency = latency

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


def worker(
    worker_id: int,
    url: str,
    features_pool: List[List[float]],
    end_time: float,
    stats: LoadStats,
    timeout: float = 5.0,
) -> None:
    """Worker loop that sends requests until end_time."""
    session = requests.Session()
    n = len(features_pool)
    rng = random.Random(worker_id + int(time.time()))

    while time.perf_counter() < end_time:
        idx = rng.randrange(n)
        payload = {"features": features_pool[idx]}
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
    p = argparse.ArgumentParser(description="Load-test the classifier /predict endpoint.")
    p.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="URL of the classifier /predict endpoint.",
    )
    p.add_argument(
        "--duration-seconds",
        type=int,
        default=60,
        help="Duration of load test (in seconds).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent worker threads.",
    )
    p.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="MNIST samples to use in the request pool.",
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

    print(f"[load_test_classifier] Target URL: {args.url}")
    print(
        f"[load_test_classifier] duration={args.duration_seconds}s, "
        f"concurrency={args.concurrency}, dataset_size={args.dataset_size}"
    )

    features_pool, labels = load_mnist_subset(args.dataset_size)
    print(f"[load_test_classifier] Example label distribution (first 10): {labels[:10]}")

    stats = LoadStats()
    end_time = time.perf_counter() + args.duration_seconds

    threads = []
    for wid in range(args.concurrency):
        t = threading.Thread(
            target=worker,
            args=(wid, args.url, features_pool, end_time, stats),
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
                    f"[load_test_classifier] t={elapsed:5.1f}s | "
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
        print("\n[load_test_classifier] Interrupted by user; waiting for workers to finish...")

    # Wait for all threads to exit
    for t in threads:
        t.join(timeout=1.0)

    # Final stats
    elapsed = time.perf_counter() - start_time
    snap = stats.snapshot()
    rps = snap["total_requests"] / elapsed if elapsed > 0 else 0.0

    print("\n[load_test_classifier] FINAL STATS")
    print(f"  elapsed:       {elapsed:.1f} s")
    print(f"  requests:      {snap['total_requests']}")
    print(f"  successes:     {snap['total_success']}")
    print(f"  errors:        {snap['total_errors']}")
    print(f"  RPS:           {rps:.1f}")
    print(f"  avg latency:   {snap['avg_latency']*1000:.1f} ms")
    print(f"  max latency:   {snap['max_latency']*1000:.1f} ms")


if __name__ == "__main__":
    main()

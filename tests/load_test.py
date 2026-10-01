"""
load_test.py
------------
Phase 7 — Performance & Load Testing

Two modes:
  1. Locust web UI  →  locust -f tests/load_test.py --host http://localhost:8000
  2. Direct script  →  python tests/load_test.py --host http://localhost:8000

The direct script runs each experiment sequentially and saves:
  results/latency_results.csv
  results/graphs/  (PNG charts)
"""

import argparse
import csv
import os
import time
import statistics
import concurrent.futures

import requests
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import matplotlib.pyplot as plt

# ── Sample input (malignant, sample #0 from Breast Cancer dataset) ────────────
SAMPLE_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

REQUEST_BODY = {"features": SAMPLE_FEATURES}

# ── Endpoints to test ─────────────────────────────────────────────────────────
ENDPOINTS = {
    "baseline":  "/predict",
    "shap":      "/explain/shap",
    "lime":      "/explain/lime",
    "combined":  "/explain",
}

# ── Load volumes ──────────────────────────────────────────────────────────────
LOAD_LEVELS = [10, 50, 100, 500, 1000]


def single_request(host: str, endpoint: str) -> dict:
    """Send one POST request and return timing info."""
    url = f"{host}{endpoint}"
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=REQUEST_BODY, timeout=60)
        t1 = time.perf_counter()
        return {
            "success": resp.status_code == 200,
            "latency_ms": round((t1 - t0) * 1000, 2),
            "status_code": resp.status_code,
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "success": False,
            "latency_ms": round((t1 - t0) * 1000, 2),
            "status_code": 0,
            "error": str(e),
        }


def run_load_test(host: str, endpoint: str, n_requests: int,
                  max_workers: int = 5) -> dict:
    """
    Run n_requests in parallel (max_workers at a time) and return stats.
    max_workers=5 is conservative — keeps Minikube single pod from being
    overwhelmed while still testing concurrency.
    """
    results = []
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(single_request, host, endpoint)
                   for _ in range(n_requests)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    t_end = time.perf_counter()
    wall_time = t_end - t_start

    latencies = [r["latency_ms"] for r in results if r["success"]]
    successes = sum(1 for r in results if r["success"])
    failures  = n_requests - successes

    if not latencies:
        latencies = [0]

    return {
        "n_requests":   n_requests,
        "successes":    successes,
        "failures":     failures,
        "avg_ms":       round(statistics.mean(latencies), 2),
        "median_ms":    round(statistics.median(latencies), 2),
        "min_ms":       round(min(latencies), 2),
        "max_ms":       round(max(latencies), 2),
        "throughput":   round(successes / wall_time, 2),
        "wall_time_s":  round(wall_time, 2),
    }


def save_csv(all_results: list, output_path: str):
    fieldnames = [
        "endpoint", "n_requests", "successes", "failures",
        "avg_ms", "median_ms", "min_ms", "max_ms",
        "throughput", "wall_time_s"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved → {output_path}")


def plot_results(all_results: list, graphs_dir: str):
    os.makedirs(graphs_dir, exist_ok=True)

    # Group by endpoint
    by_endpoint = {}
    for r in all_results:
        ep = r["endpoint"]
        if ep not in by_endpoint:
            by_endpoint[ep] = {"loads": [], "avg_ms": [], "throughput": []}
        by_endpoint[ep]["loads"].append(r["n_requests"])
        by_endpoint[ep]["avg_ms"].append(r["avg_ms"])
        by_endpoint[ep]["throughput"].append(r["throughput"])

    colors = {
        "baseline": "#6366f1",
        "shap":     "#8b5cf6",
        "lime":     "#f59e0b",
        "combined": "#059669",
    }

    # ── Graph 1: Load vs Avg Latency ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for ep, data in by_endpoint.items():
        ax.plot(data["loads"], data["avg_ms"],
                marker="o", label=ep, color=colors.get(ep))
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Request Load vs Average Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "latency_vs_load.png"), dpi=150)
    plt.close()
    print("Saved: latency_vs_load.png")

    # ── Graph 2: Load vs Throughput ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for ep, data in by_endpoint.items():
        ax.plot(data["loads"], data["throughput"],
                marker="s", label=ep, color=colors.get(ep))
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Throughput (requests/sec)")
    ax.set_title("Request Load vs Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "throughput_vs_load.png"), dpi=150)
    plt.close()
    print("Saved: throughput_vs_load.png")

    # ── Graph 3: Baseline vs SHAP vs LIME (bar chart at 100 requests) ─────────
    target_n = 100
    bar_data = {}
    for r in all_results:
        if r["n_requests"] == target_n:
            bar_data[r["endpoint"]] = r["avg_ms"]

    if bar_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(bar_data.keys())
        values = list(bar_data.values())
        bar_colors = [colors.get(l, "#888") for l in labels]
        bars = ax.bar(labels, values, color=bar_colors, width=0.5)
        ax.bar_label(bars, fmt="%.1f ms", padding=3)
        ax.set_xlabel("Endpoint")
        ax.set_ylabel("Average Latency (ms)")
        ax.set_title(f"Latency Comparison at {target_n} Requests")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "latency_comparison.png"), dpi=150)
        plt.close()
        print("Saved: latency_comparison.png")

    # ── Graph 4: Overhead vs Baseline ─────────────────────────────────────────
    baseline_by_load = {
        r["n_requests"]: r["avg_ms"]
        for r in all_results if r["endpoint"] == "baseline"
    }
    if baseline_by_load:
        fig, ax = plt.subplots(figsize=(10, 5))
        for ep in ["shap", "lime", "combined"]:
            if ep not in by_endpoint:
                continue
            overheads = []
            loads = by_endpoint[ep]["loads"]
            for load, avg in zip(loads, by_endpoint[ep]["avg_ms"]):
                base = baseline_by_load.get(load, 1)
                overheads.append(round((avg - base) / base * 100, 1))
            ax.plot(loads, overheads, marker="o",
                    label=f"{ep} overhead %", color=colors.get(ep))
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel("XAI Overhead (%)")
        ax.set_title("XAI Overhead % vs Request Load")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, "overhead_vs_load.png"), dpi=150)
        plt.close()
        print("Saved: overhead_vs_load.png")


def main():
    parser = argparse.ArgumentParser(description="XAI Load Tester")
    parser.add_argument("--host", default="http://localhost:8000",
                        help="API base URL")
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent workers per test")
    parser.add_argument("--endpoints", nargs="+",
                        default=list(ENDPOINTS.keys()),
                        choices=list(ENDPOINTS.keys()),
                        help="Endpoints to test")
    parser.add_argument("--loads", nargs="+", type=int,
                        default=LOAD_LEVELS,
                        help="Request volumes to test")
    args = parser.parse_args()

    print(f"XAI Load Test — Target: {args.host}")
    print(f"Endpoints: {args.endpoints}")
    print(f"Load levels: {args.loads}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    # Verify the API is reachable
    try:
        r = requests.get(f"{args.host}/health", timeout=10)
        assert r.json()["status"] == "healthy"
        print("API health check: OK\n")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.host} — {e}")
        print("Make sure the server is running before running load tests.")
        return

    all_results = []

    for ep_name in args.endpoints:
        endpoint = ENDPOINTS[ep_name]
        print(f"\n── Testing: {ep_name} ({endpoint}) ──────────────────────")
        for n in args.loads:
            print(f"  {n:>5} requests ... ", end="", flush=True)
            stats = run_load_test(args.host, endpoint, n, args.workers)
            stats["endpoint"] = ep_name
            all_results.append(stats)
            print(
                f"avg={stats['avg_ms']}ms  "
                f"median={stats['median_ms']}ms  "
                f"throughput={stats['throughput']} req/s  "
                f"failures={stats['failures']}"
            )

    # Save results
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "latency_results.csv")
    save_csv(all_results, csv_path)

    graphs_dir = os.path.join("results", "graphs")
    plot_results(all_results, graphs_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Endpoint':<12} {'Requests':>8} {'Avg ms':>8} {'Median':>8} "
          f"{'Min':>8} {'Max':>8} {'Req/s':>8} {'Fail':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['endpoint']:<12} {r['n_requests']:>8} {r['avg_ms']:>8} "
              f"{r['median_ms']:>8} {r['min_ms']:>8} {r['max_ms']:>8} "
              f"{r['throughput']:>8} {r['failures']:>6}")

    print("\nDone. Results saved to results/")


if __name__ == "__main__":
    main()


# ── Locust tasks (used when running: locust -f tests/load_test.py) ────────────
try:
    from locust import HttpUser, task, between

    class XAIUser(HttpUser):
        wait_time = between(1, 3)

        @task(4)
        def predict(self):
            self.client.post("/predict", json=REQUEST_BODY)

        @task(2)
        def explain_shap(self):
            self.client.post("/explain/shap", json=REQUEST_BODY)

        @task(2)
        def explain_lime(self):
            self.client.post("/explain/lime", json=REQUEST_BODY)

        @task(1)
        def explain_combined(self):
            self.client.post("/explain", json=REQUEST_BODY)

except ImportError:
    pass  # locust not installed — direct script mode only

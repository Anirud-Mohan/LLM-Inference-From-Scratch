"""Scrape vLLM's Prometheus /metrics endpoint during a load test.

vLLM exposes inference metrics at <pod_url>/metrics in Prometheus text format.
Run this script in a separate terminal while Locust is running. It polls
every INTERVAL seconds and writes a CSV snapshot so you can track metrics
over time.

Key metrics captured:
  - TTFT  (Time to First Token)  — vllm:time_to_first_token_seconds
  - ITL   (Inter-Token Latency)  — vllm:time_per_output_token_seconds
  - Throughput (tokens/sec)      — vllm:avg_generation_throughput_toks_per_s
  - Request throughput (req/sec) — vllm:avg_prompt_throughput_toks_per_s
  - GPU KV cache usage (%)       — vllm:gpu_cache_usage_perc
  - Requests running             — vllm:num_requests_running
  - Requests waiting             — vllm:num_requests_waiting

Usage:
    # While Locust is running in another terminal:
    python collect_metrics.py \
        --pod-url https://<POD_ID>-8000.proxy.runpod.net \
        --interval 10 \
        --output results/vllm_metrics.csv

    # For two pods (runs collection for both):
    python collect_metrics.py \
        --pod-url https://<POD1>-8000.proxy.runpod.net \
                  https://<POD2>-8000.proxy.runpod.net \
        --interval 10 \
        --output results/vllm_metrics.csv
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Prometheus metric names exposed by vLLM
GAUGE_METRICS = {
    "ttft_p50":         'vllm:time_to_first_token_seconds',
    "itl_p50":          'vllm:time_per_output_token_seconds',
    "gen_throughput":   'vllm:avg_generation_throughput_toks_per_s',
    "prompt_throughput":'vllm:avg_prompt_throughput_toks_per_s',
    "gpu_cache_pct":    'vllm:gpu_cache_usage_perc',
    "requests_running": 'vllm:num_requests_running',
    "requests_waiting": 'vllm:num_requests_waiting',
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Extract scalar gauge values from Prometheus text format."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0].split("{")[0]  # strip label selectors
            try:
                values[metric_name] = float(parts[-1])
            except ValueError:
                pass
    return values


def scrape_once(pod_url: str) -> dict[str, float]:
    """Fetch /metrics from one pod and return relevant values."""
    url = f"{pod_url.rstrip('/')}/metrics"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw = parse_prometheus(resp.text)
        result: dict[str, float] = {}
        for col, prom_name in GAUGE_METRICS.items():
            result[col] = raw.get(prom_name, float("nan"))
        return result
    except Exception as e:
        print(f"  [WARNING] Could not scrape {url}: {e}")
        return {col: float("nan") for col in GAUGE_METRICS}


def main():
    parser = argparse.ArgumentParser(description="vLLM metrics scraper")
    parser.add_argument(
        "--pod-url", nargs="+", required=True,
        help="One or more pod proxy URLs"
    )
    parser.add_argument(
        "--interval", type=int, default=10,
        help="Scrape interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--output", default="results/vllm_metrics.csv",
        help="Output CSV path (default: results/vllm_metrics.csv)"
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = out_path.open("a", newline="")
    writer = csv.writer(csv_file)

    header = ["timestamp", "pod_url"] + list(GAUGE_METRICS.keys())
    if out_path.stat().st_size == 0:
        writer.writerow(header)

    print(f"[Metrics] Scraping {len(args.pod_url)} pod(s) every {args.interval}s")
    print(f"[Metrics] Writing to {out_path}")
    print("[Metrics] Press Ctrl+C to stop\n")

    def _shutdown(sig, frame):
        print("\n[Metrics] Stopping.")
        csv_file.flush()
        csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for pod_url in args.pod_url:
            metrics = scrape_once(pod_url)
            row = [ts, pod_url] + [metrics[col] for col in GAUGE_METRICS]
            writer.writerow(row)

            ttft = metrics.get("ttft_p50", float("nan"))
            itl = metrics.get("itl_p50", float("nan"))
            gen_tp = metrics.get("gen_throughput", float("nan"))
            running = metrics.get("requests_running", float("nan"))
            waiting = metrics.get("requests_waiting", float("nan"))
            cache = metrics.get("gpu_cache_pct", float("nan"))

            print(
                f"[{ts}] {pod_url.split('//')[1][:30]}  "
                f"TTFT={ttft*1000:.0f}ms  ITL={itl*1000:.1f}ms  "
                f"throughput={gen_tp:.1f}tok/s  "
                f"running={running:.0f}  waiting={waiting:.0f}  "
                f"kv_cache={cache*100:.1f}%"
            )

        csv_file.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

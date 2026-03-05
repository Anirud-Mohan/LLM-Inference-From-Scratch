"""GPU utilization monitor — works with both pipelines.

For the custom pipeline : polls GET <pod_url>/gpu_stats  (JSON)
For the vLLM pipeline   : polls GET <pod_url>/metrics    (Prometheus text)

Run alongside Locust in a separate terminal:

    # Custom pipeline:
    python gpu_monitor.py --run-tag custom \
        --pod-url https://<POD1>-8000.proxy.runpod.net \
                  https://<POD2>-8000.proxy.runpod.net

    # vLLM pipeline:
    python gpu_monitor.py --run-tag vllm \
        --pod-url https://<POD1>-8000.proxy.runpod.net \
                  https://<POD2>-8000.proxy.runpod.net

Output: results/gpu_metrics_<run_tag>.csv
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

VLLM_GPU_METRIC = "vllm:gpu_cache_usage_perc"


def _scrape_custom(pod_url: str) -> dict:
    """Fetch /gpu_stats from the custom FastAPI server."""
    url = f"{pod_url.rstrip('/')}/gpu_stats"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "gpu_util_pct":  data.get("gpu_util_pct"),
        "vram_used_mb":  data.get("vram_used_mb"),
        "vram_total_mb": data.get("vram_total_mb"),
    }


def _scrape_vllm(pod_url: str) -> dict:
    """Parse vLLM's Prometheus /metrics for GPU cache usage."""
    url = f"{pod_url.rstrip('/')}/metrics"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    gpu_cache_pct = None
    for line in resp.text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].split("{")[0] == VLLM_GPU_METRIC:
            try:
                gpu_cache_pct = float(parts[-1]) * 100
            except ValueError:
                pass
    return {
        "gpu_util_pct":  gpu_cache_pct,
        "vram_used_mb":  None,
        "vram_total_mb": None,
    }


def scrape(pod_url: str, run_tag: str) -> dict:
    if run_tag == "vllm":
        return _scrape_vllm(pod_url)
    return _scrape_custom(pod_url)


def main():
    parser = argparse.ArgumentParser(description="GPU metrics collector")
    parser.add_argument("--pod-url", nargs="+", required=True,
                        help="One or more pod proxy URLs")
    parser.add_argument("--run-tag", required=True, choices=["custom", "vllm"],
                        help="Pipeline label: 'custom' or 'vllm'")
    parser.add_argument("--interval", type=int, default=10,
                        help="Poll interval in seconds (default: 10)")
    args = parser.parse_args()

    out_path = Path(f"results/gpu_metrics_{args.run_tag}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = out_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "pod_url", "gpu_util_pct", "vram_used_mb", "vram_total_mb"])

    print(f"[GPU Monitor] run_tag={args.run_tag}  pods={len(args.pod_url)}  interval={args.interval}s")
    print(f"[GPU Monitor] Writing to {out_path}")
    print("[GPU Monitor] Press Ctrl+C to stop\n")

    def _shutdown(sig, frame):
        print("\n[GPU Monitor] Stopped.")
        csv_file.flush()
        csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for pod_url in args.pod_url:
            try:
                m = scrape(pod_url, args.run_tag)
                writer.writerow([
                    ts, pod_url,
                    m["gpu_util_pct"], m["vram_used_mb"], m["vram_total_mb"],
                ])
                csv_file.flush()
                util_str  = f"{m['gpu_util_pct']:.1f}%" if m["gpu_util_pct"] is not None else "n/a"
                vram_str  = (
                    f"{m['vram_used_mb']:.0f}/{m['vram_total_mb']:.0f} MB"
                    if m["vram_used_mb"] is not None else "n/a"
                )
                print(f"[{ts}] {pod_url.split('//')[-1][:35]}  gpu_util={util_str}  vram={vram_str}")
            except Exception as e:
                print(f"[{ts}] WARNING — could not scrape {pod_url}: {e}")
                writer.writerow([ts, pod_url, None, None, None])
                csv_file.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

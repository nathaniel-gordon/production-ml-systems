"""Real-Time Latency Percentiles & SLA Violation Monitor.

Simulates low-latency serving metrics:
- P50, P90, P95, P99, P99.9 latency percentiles
- SLA violation detection (e.g. < 25ms threshold)
- Dynamic batching queuing latency analysis
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LatencyStats:
    total_requests: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    sla_violations: int
    sla_compliance_pct: float


class LatencyTracker:
    def __init__(self, sla_threshold_ms: float = 25.0):
        self.sla_threshold_ms = sla_threshold_ms
        self.latencies_ms: List[float] = []

    def record_latency(self, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))

    def compute_stats(self) -> LatencyStats:
        if not self.latencies_ms:
            return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 100.0)

        arr = np.array(self.latencies_ms)
        violations = int(np.sum(arr > self.sla_threshold_ms))
        compliance = (1.0 - (violations / len(arr))) * 100.0

        return LatencyStats(
            total_requests=len(arr),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            p999_ms=float(np.percentile(arr, 99.9)),
            mean_ms=float(np.mean(arr)),
            sla_violations=violations,
            sla_compliance_pct=round(compliance, 2),
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Latency & SLA Monitor")
    parser.add_argument("--requests", type=int, default=5000, help="Number of simulated requests")
    parser.add_argument("--sla", type=float, default=25.0, help="SLA threshold in ms")
    args = parser.parse_args()

    print("================================================================================")
    print(f" ⏱️ Real-Time Inference Latency & SLA Monitor (Threshold = {args.sla} ms)")
    print("================================================================================")

    np.random.seed(42)
    tracker = LatencyTracker(sla_threshold_ms=args.sla)

    # Simulate realistic serving latency: log-normal distribution with long tail
    simulated_latencies = np.random.lognormal(mean=2.2, sigma=0.45, size=args.requests)
    for lat in simulated_latencies:
        tracker.record_latency(lat)

    stats = tracker.compute_stats()

    print(f"Total Requests Evaluated: {stats.total_requests:,}")
    print(f"Mean Latency:             {stats.mean_ms:.2f} ms")
    print(f"P50 (Median):             {stats.p50_ms:.2f} ms")
    print(f"P90 Latency:              {stats.p90_ms:.2f} ms")
    print(f"P95 Latency:              {stats.p95_ms:.2f} ms")
    print(f"P99 Latency:              {stats.p99_ms:.2f} ms")
    print(f"P99.9 (Tail Spike):       {stats.p999_ms:.2f} ms")
    print("--------------------------------------------------------------------------------")
    print(f"SLA Violations (> {args.sla}ms): {stats.sla_violations:,} ({stats.sla_compliance_pct:.2f}% Compliance)")
    print("================================================================================\n")


if __name__ == "__main__":
    main()

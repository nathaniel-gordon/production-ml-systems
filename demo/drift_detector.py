"""Statistical Data Drift & Covariate Shift Monitoring Engine.

Implements production-grade statistical drift detection metrics:
- Population Stability Index (PSI) with adaptive binning
- Kolmogorov-Smirnov (KS) two-sample non-parametric test
- Wasserstein Distance (Earth Mover's Distance)
- Severity-graded alerting (Green / Yellow / Red)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    feature_name: str
    psi: float
    psi_status: str  # "STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    ks_statistic: float
    ks_pvalue: float
    ks_drift_detected: bool
    wasserstein_distance: float


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI).

    Formula:
        PSI = sum( (Actual_pct_i - Expected_pct_i) * ln(Actual_pct_i / Expected_pct_i) )

    Interpretation:
        PSI < 0.10: No significant distribution change (Stable)
        0.10 <= PSI < 0.25: Moderate change (Warning / Monitor)
        PSI >= 0.25: Significant distribution shift (Action Required / Retrain)
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantiles based on reference/expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # Ensure unique bin edges
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    # Calculate frequency counts
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Convert to proportions with smoothing epsilon
    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    # Normalize after adding epsilon
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # PSI calculation
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def calculate_ks_drift(
    expected: np.ndarray,
    actual: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, bool]:
    """Perform two-sample Kolmogorov-Smirnov test for continuous distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0, False

    ks_res = stats.ks_2samp(expected, actual)
    statistic = float(ks_res.statistic)
    pvalue = float(ks_res.pvalue)
    drift_detected = pvalue < alpha
    return statistic, pvalue, drift_detected


class DriftDetector:
    def __init__(self, psi_warning_threshold: float = 0.10, psi_critical_threshold: float = 0.25):
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold

    def evaluate_feature(self, feature_name: str, reference_data: np.ndarray, production_data: np.ndarray) -> DriftReport:
        psi = calculate_psi(reference_data, production_data)
        ks_stat, ks_pval, ks_drift = calculate_ks_drift(reference_data, production_data)
        w_dist = float(stats.wasserstein_distance(reference_data, production_data))

        if psi < self.psi_warning_threshold:
            status = "STABLE"
        elif psi < self.psi_critical_threshold:
            status = "MODERATE_DRIFT"
        else:
            status = "SIGNIFICANT_DRIFT"

        return DriftReport(
            feature_name=feature_name,
            psi=round(psi, 4),
            psi_status=status,
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 6),
            ks_drift_detected=ks_drift,
            wasserstein_distance=round(w_dist, 4),
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="ML Drift & Covariate Shift Monitor")
    parser.add_argument("--samples", type=int, default=2000, help="Number of baseline samples")
    args = parser.parse_args()

    print("================================================================================")
    print(" 📊 Production ML Drift & Covariate Shift Analyzer")
    print("================================================================================")

    np.random.seed(42)
    detector = DriftDetector()

    # Generate synthetic features with varying drift levels
    # 1. Stable feature (Standard Normal)
    f1_ref = np.random.normal(0.0, 1.0, args.samples)
    f1_prod = np.random.normal(0.02, 1.01, args.samples)

    # 2. Moderate drift (Slight mean & variance shift)
    f2_ref = np.random.gamma(shape=2.0, scale=1.5, size=args.samples)
    f2_prod = np.random.gamma(shape=2.4, scale=1.7, size=args.samples)

    # 3. Severe drift (Significant covariate shift / anomaly)
    f3_ref = np.random.beta(a=2, b=5, size=args.samples)
    f3_prod = np.random.beta(a=5, b=2, size=args.samples)

    features = [
        ("user_transaction_amount", f1_ref, f1_prod),
        ("session_duration_minutes", f2_ref, f2_prod),
        ("account_risk_score", f3_ref, f3_prod),
    ]

    print(f"{'Feature Name':<28} | {'PSI':<8} | {'PSI Status':<18} | {'KS Stat':<8} | {'KS p-val':<10} | {'W-Dist':<8}")
    print("-" * 92)

    for name, ref_data, prod_data in features:
        rep = detector.evaluate_feature(name, ref_data, prod_data)
        indicator = "🟢" if rep.psi_status == "STABLE" else "🟡" if rep.psi_status == "MODERATE_DRIFT" else "🔴"
        print(f"{rep.feature_name:<28} | {rep.psi:<8.4f} | {indicator + ' ' + rep.psi_status:<18} | {rep.ks_statistic:<8.4f} | {rep.ks_pvalue:<10.4e} | {rep.wasserstein_distance:<8.4f}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()

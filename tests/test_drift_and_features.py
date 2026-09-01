import datetime
import numpy as np
import pytest
from demo.drift_detector import DriftDetector, calculate_psi, calculate_ks_drift
from demo.feature_store_simulator import FeatureStore
from demo.latency_monitor import LatencyTracker


def test_psi_zero_for_identical_distributions():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    psi = calculate_psi(data, data)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_detects_distribution_shift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 2000)
    prod = np.random.normal(2.5, 1, 2000)  # strong shift
    psi = calculate_psi(ref, prod)
    assert psi > 0.25  # Should trigger SIGNIFICANT_DRIFT


def test_ks_test_drift():
    ref = np.random.normal(0, 1, 1000)
    prod = np.random.normal(1.5, 1, 1000)
    stat, pval, drift = calculate_ks_drift(ref, prod)
    assert drift is True
    assert pval < 0.05


def test_feature_store_point_in_time_correctness():
    store = FeatureStore()
    t0 = datetime.datetime(2026, 1, 1, 12, 0, 0)

    store.ingest_record("cust_1", "income", 50000, t0)
    store.ingest_record("cust_1", "income", 80000, t0 + datetime.timedelta(days=30))

    # Observation at day 15 (should see 50,000, NOT 80,000)
    obs = [{"entity_id": "cust_1", "event_timestamp": t0 + datetime.timedelta(days=15)}]
    res = store.get_historical_features(obs, ["income"])
    assert res[0]["income"] == 50000

    # Observation at day 45 (should see 80,000)
    obs2 = [{"entity_id": "cust_1", "event_timestamp": t0 + datetime.timedelta(days=45)}]
    res2 = store.get_historical_features(obs2, ["income"])
    assert res2[0]["income"] == 80000


def test_latency_tracker_percentiles():
    tracker = LatencyTracker(sla_threshold_ms=50.0)
    for i in range(1, 101):
        tracker.record_latency(i)

    stats = tracker.compute_stats()
    assert stats.total_requests == 100
    assert stats.p50_ms == pytest.approx(50.5, rel=1e-1)
    assert stats.p99_ms == pytest.approx(99.0, rel=1e-1)
    assert stats.sla_violations == 50

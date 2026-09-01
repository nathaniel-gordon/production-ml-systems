"""Dual-Storage Feature Store Simulator with Point-in-Time Correctness.

Simulates enterprise feature stores (Feast / Hopsworks / Tecton):
- Low-latency Online Store (Redis / Key-Value) for real-time serving
- Immutable Offline Store (Parquet / Time-Series) for training set generation
- Point-in-Time Correctness (As-Of Joins) preventing data leakage
"""
from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FeatureRecord:
    entity_id: str
    feature_name: str
    feature_value: Any
    timestamp: datetime.datetime


@dataclass
class FeatureView:
    name: str
    entities: List[str]
    features: List[str]


class FeatureStore:
    def __init__(self):
        # Offline store: immutable append-only records
        self._offline_records: List[FeatureRecord] = []
        # Online store: entity_id -> { feature_name: (value, timestamp) }
        self._online_store: Dict[str, Dict[str, Tuple[Any, datetime.datetime]]] = {}

    def ingest_record(self, entity_id: str, feature_name: str, value: Any, timestamp: datetime.datetime) -> None:
        """Ingest a feature value into both online and offline layers."""
        record = FeatureRecord(
            entity_id=entity_id,
            feature_name=feature_name,
            feature_value=value,
            timestamp=timestamp,
        )
        self._offline_records.append(record)

        # Update online store if timestamp is newer
        current = self._online_store.setdefault(entity_id, {}).get(feature_name)
        if current is None or timestamp >= current[1]:
            self._online_store[entity_id][feature_name] = (value, timestamp)

    def get_online_features(self, entity_ids: List[str], feature_names: List[str]) -> List[Dict[str, Any]]:
        """Retrieve low-latency online features for real-time inference."""
        results = []
        for e_id in entity_ids:
            row = {"entity_id": e_id}
            entity_features = self._online_store.get(e_id, {})
            for f_name in feature_names:
                val_tuple = entity_features.get(f_name)
                row[f_name] = val_tuple[0] if val_tuple else None
            results.append(row)
        return results

    def get_historical_features(
        self,
        observation_events: List[Dict[str, Any]],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Perform zero-leakage Point-in-Time (As-Of) Join.

        Each observation event must contain:
        - "entity_id": identifier
        - "event_timestamp": datetime of observation
        """
        training_rows = []
        # Sort offline records chronologically for deterministic as-of search
        sorted_records = sorted(self._offline_records, key=lambda r: r.timestamp)

        for obs in observation_events:
            e_id = obs["entity_id"]
            obs_time = obs["event_timestamp"]
            row = dict(obs)

            for f_name in feature_names:
                # Find most recent record with timestamp <= obs_time
                valid_records = [
                    r for r in sorted_records
                    if r.entity_id == e_id and r.feature_name == f_name and r.timestamp <= obs_time
                ]
                if valid_records:
                    row[f_name] = valid_records[-1].feature_value
                else:
                    row[f_name] = None

            training_rows.append(row)
        return training_rows


# Alias for backward compatibility
PointInTimeJoiner = FeatureStore


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("================================================================================")
    print(" 🏪 Dual-Storage Feature Store & Point-in-Time Correctness Simulator")
    print("================================================================================")

    store = FeatureStore()
    base_time = datetime.datetime(2026, 1, 1, 10, 0, 0)

    # Ingest event timeline for customer "user_101"
    # Event 1: Initial credit score at 10:00 AM
    store.ingest_record("user_101", "credit_score", 680, base_time)
    store.ingest_record("user_101", "fraud_risk_score", 0.05, base_time)

    # Event 2: Credit score updated at 12:00 PM
    store.ingest_record("user_101", "credit_score", 720, base_time + datetime.timedelta(hours=2))

    # Event 3: High risk transaction at 2:00 PM
    store.ingest_record("user_101", "fraud_risk_score", 0.88, base_time + datetime.timedelta(hours=4))

    # 1. Online Real-Time Serving Query
    print("\n1. Online Store Query (Real-Time Current State):")
    online_res = store.get_online_features(["user_101"], ["credit_score", "fraud_risk_score"])
    print(f"   {online_res}")

    # 2. Historical Training Set Generation (As-Of Join)
    print("\n2. Offline Point-in-Time As-Of Joins (Zero Data Leakage):")
    observations = [
        # Observation at 10:30 AM (Should see credit_score=680, fraud_risk_score=0.05)
        {"entity_id": "user_101", "event_timestamp": base_time + datetime.timedelta(minutes=30), "label_loan_approved": 1},
        # Observation at 1:00 PM (Should see credit_score=720, fraud_risk_score=0.05 -- NOT the 2:00 PM 0.88!)
        {"entity_id": "user_101", "event_timestamp": base_time + datetime.timedelta(hours=3), "label_loan_approved": 1},
    ]

    training_data = store.get_historical_features(observations, ["credit_score", "fraud_risk_score"])
    for i, row in enumerate(training_data, 1):
        print(f"   Observation #{i} @ {row['event_timestamp'].strftime('%H:%M:%S')}: "
              f"credit_score={row['credit_score']}, fraud_risk={row['fraud_risk_score']}, target={row['label_loan_approved']}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()

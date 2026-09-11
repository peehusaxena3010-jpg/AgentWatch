"""
Trains an Isolation Forest anomaly detection model on stored traces, and
writes the resulting anomaly_score back into the database for each trace.

Isolation Forest works by randomly partitioning the feature space; points
that get isolated in fewer splits (i.e. sit apart from the bulk of "normal"
traces) get a higher anomaly score. It's unsupervised -- it does NOT need
labels -- which matters here because you won't have a large hand-labeled
dataset early on. As you accumulate confirmed human reviews later (Phase 4,
the feedback loop), those labels can be used to validate/calibrate this
model's threshold, but the model itself doesn't require them to train.

Run with:
    python ml/train_anomaly_model.py
"""

from __future__ import annotations

import math
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.database import SessionLocal, TraceRecord
from ml.feature_extraction import trace_to_features, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "anomaly_model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")


def load_traces_as_dataframe() -> tuple[pd.DataFrame, list[str]]:
    """Reads all stored traces from the DB and extracts features into a DataFrame."""
    db = SessionLocal()
    try:
        records = db.query(TraceRecord).all()
        trace_ids = []
        feature_rows = []
        for record in records:
            trace_dict = json.loads(record.full_trace_json)
            feature_rows.append(trace_to_features(trace_dict))
            trace_ids.append(record.trace_id)
        df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        return df, trace_ids
    finally:
        db.close()


def train_and_score() -> dict[str, Any]:
    df, trace_ids = load_traces_as_dataframe()

    if len(df) < 5:
        msg = f"Only {len(df)} traces found. Isolation Forest needs at least 5 traces (aim for 20+)."
        print(f"⚠️  {msg}")
        if len(df) == 0:
            return {"status": "error", "message": msg, "count": 0, "flagged": 0}

    print(f"Training on {len(df)} traces with {len(FEATURE_NAMES)} features...")

    # Log-transform latency features to prevent extreme latency spikes from dominating
    df_transformed = df.copy()
    for col in ["total_latency_ms", "avg_step_latency_ms", "max_step_latency_ms"]:
        df_transformed[col] = df_transformed[col].apply(math.log1p)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=42,
    )
    model.fit(df_transformed)

    raw_scores = model.decision_function(df_transformed)
    min_score, max_score = float(raw_scores.min()), float(raw_scores.max())
    denominator = (max_score - min_score) if (max_score - min_score) > 1e-9 else 1.0
    anomaly_scores = 1 - (raw_scores - min_score) / denominator

    predictions = model.predict(df_transformed)  # -1 = anomaly, 1 = normal

    # Save model and normalization metadata
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "min_raw_score": min_score,
        "max_raw_score": max_score,
        "n_samples": len(df),
        "features": FEATURE_NAMES,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved to {MODEL_PATH}")

    # Write scores back into database
    db = SessionLocal()
    flagged_count = 0
    try:
        for trace_id, score, pred in zip(trace_ids, anomaly_scores, predictions):
            record = db.get(TraceRecord, trace_id)
            if record is None:
                continue
            record.anomaly_score = float(score)

            full_trace = json.loads(record.full_trace_json)
            full_trace["anomaly_score"] = float(score)
            record.full_trace_json = json.dumps(full_trace)

            if pred == -1:
                flagged_count += 1
        db.commit()
    finally:
        db.close()

    print(f"✅ Scored {len(trace_ids)} traces. {flagged_count} flagged as anomalous by Isolation Forest.")

    return {
        "status": "success",
        "total_scored": len(trace_ids),
        "flagged_count": flagged_count,
        "model_path": MODEL_PATH,
    }


def predict_single_trace(trace: dict[str, Any]) -> float:
    """
    Infers the anomaly score for a single trace dict in real-time.
    Falls back to feature heuristic if model is not yet saved.
    """
    feats = trace_to_features(trace)

    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            df = pd.DataFrame([feats], columns=FEATURE_NAMES)
            for col in ["total_latency_ms", "avg_step_latency_ms", "max_step_latency_ms"]:
                df[col] = df[col].apply(math.log1p)

            raw = float(model.decision_function(df)[0])
            min_s, max_s = meta["min_raw_score"], meta["max_raw_score"]
            denom = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0
            score = 1.0 - (raw - min_s) / denom
            return max(0.0, min(1.0, round(score, 3)))
        except Exception:
            pass

    # Heuristic fallback if model not fitted yet
    penalty = 0.0
    if feats.get("is_error_status"):
        penalty += 0.35
    if feats.get("repeated_consecutive_calls", 0) > 0:
        penalty += 0.4
    if feats.get("num_failed_steps", 0) > 0:
        penalty += 0.25
    if feats.get("total_latency_ms", 0) > 10000:
        penalty += 0.2
    return max(0.0, min(1.0, round(penalty, 3)))


if __name__ == "__main__":
    train_and_score()
"""
Data access and pipeline orchestration layer for the AgentWatch dashboard.
Kept cleanly separated from the Streamlit UI components.
"""

from __future__ import annotations

import json
import os
import sys
import pandas as pd
from typing import Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.database import SessionLocal, TraceRecord
from ml.threshold_config import load_threshold


def load_traces_df() -> pd.DataFrame:
    """Loads all traces from SQLite into a structured pandas DataFrame."""
    db = SessionLocal()
    try:
        records = db.query(TraceRecord).order_by(TraceRecord.timestamp.desc()).all()
        rows = []
        for r in records:
            try:
                full_trace = json.loads(r.full_trace_json)
            except Exception:
                full_trace = {}

            classification = full_trace.get("auditor_classification") or {}
            review = full_trace.get("human_review") or {}
            steps = full_trace.get("steps", [])

            rows.append({
                "trace_id": r.trace_id,
                "session_id": r.session_id,
                "timestamp": r.timestamp,
                "agent_name": r.agent_name,
                "agent_version": r.agent_version,
                "status": r.status,
                "user_input": full_trace.get("user_input", "")[:80],
                "num_steps": len(steps),
                "total_latency_ms": r.total_latency_ms,
                "anomaly_score": r.anomaly_score,
                "faithfulness_score": r.faithfulness_score,
                "failure_mode": classification.get("failure_mode"),
                "explanation": classification.get("explanation"),
                "reviewed": review.get("confirmed") is not None,
                "review_confirmed": review.get("confirmed"),
                "reviewer_note": review.get("reviewer_note"),
            })
        return pd.DataFrame(rows)
    finally:
        db.close()


def get_full_trace(trace_id: str) -> dict | None:
    """Fetches the complete trace dict (steps, reasoning, tool calls) by ID."""
    db = SessionLocal()
    try:
        record = db.get(TraceRecord, trace_id)
        if record is None:
            return None
        return json.loads(record.full_trace_json)
    finally:
        db.close()


def submit_human_review(trace_id: str, confirmed: bool, reviewer_note: str = "") -> bool:
    """Records human review decision (confirming a real failure vs rejecting a false positive)."""
    db = SessionLocal()
    try:
        record = db.get(TraceRecord, trace_id)
        if record is None:
            return False
        full_trace = json.loads(record.full_trace_json)
        full_trace["human_review"] = {"confirmed": confirmed, "reviewer_note": reviewer_note}
        record.full_trace_json = json.dumps(full_trace)
        db.commit()
        return True
    finally:
        db.close()


def get_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Computes headline KPIs and calibration health for dashboard summary cards."""
    if df.empty:
        return {
            "total_traces": 0,
            "success_rate": 0.0,
            "error_count": 0,
            "flagged_count": 0,
            "avg_anomaly_score": 0.0,
            "avg_faithfulness_score": 0.0,
            "reviewed_count": 0,
            "false_positive_rate": 0.0,
            "active_threshold": load_threshold(),
        }

    total = len(df)
    success_count = int((df["status"] == "success").sum())
    error_count = int((df["status"] == "error").sum())
    threshold = load_threshold()
    flagged_count = int((df["anomaly_score"] >= threshold).sum())

    avg_anomaly = float(df["anomaly_score"].dropna().mean()) if df["anomaly_score"].notna().any() else 0.0
    avg_faithfulness = float(df["faithfulness_score"].dropna().mean()) if df["faithfulness_score"].notna().any() else 0.0

    reviewed_df = df[df["reviewed"] == True]  # noqa: E712
    reviewed_count = len(reviewed_df)

    false_positive_rate = (
        float((reviewed_df["review_confirmed"] == False).mean())  # noqa: E712
        if reviewed_count > 0 else 0.0
    )

    return {
        "total_traces": total,
        "success_rate": round((success_count / total) * 100, 1),
        "error_count": error_count,
        "flagged_count": flagged_count,
        "avg_anomaly_score": round(avg_anomaly, 3),
        "avg_faithfulness_score": round(avg_faithfulness, 3),
        "reviewed_count": reviewed_count,
        "false_positive_rate": round(false_positive_rate, 3),
        "active_threshold": threshold,
    }


def execute_agent_test(user_input: str) -> dict[str, Any]:
    """Runs a live agent execution, observes it through the full pipeline, and persists it."""
    from agents.support_agent import execute_and_observe
    trace = execute_and_observe(user_input, auto_score=True, persist=True)
    return trace.model_dump()


def train_anomaly_model_action() -> dict[str, Any]:
    """Triggers Isolation Forest model retraining."""
    from ml.train_anomaly_model import train_and_score
    return train_and_score()


def score_faithfulness_action() -> int:
    """Triggers batch faithfulness scoring across all stored traces."""
    from ml.faithfulness_scorer import FaithfulnessScorer
    return FaithfulnessScorer.score_and_update_db()


def run_auditor_action() -> None:
    """Triggers batch LLM auditing on flagged traces."""
    from auditor.run_auditor import run_audit
    run_audit()


def recalibrate_threshold_action() -> dict[str, Any]:
    """Executes Youden's J threshold optimization from human reviews."""
    from ml.feedback_loop import (
        load_reviewed_traces,
        false_positive_rate_at_threshold,
        find_best_threshold,
        MIN_REVIEWS_REQUIRED,
    )
    from ml.threshold_config import load_threshold, save_threshold

    pairs = load_reviewed_traces()
    if len(pairs) < MIN_REVIEWS_REQUIRED:
        return {
            "success": False,
            "message": f"Need at least {MIN_REVIEWS_REQUIRED} reviews (currently have {len(pairs)}). Review more traces first.",
        }

    old_threshold = load_threshold()
    old_fpr, old_tpr = false_positive_rate_at_threshold(pairs, old_threshold)

    new_threshold, best_j = find_best_threshold(pairs)
    new_fpr, new_tpr = false_positive_rate_at_threshold(pairs, new_threshold)

    save_threshold(new_threshold, metadata={
        "reviewed_traces_used": len(pairs),
        "old_threshold": old_threshold,
        "old_false_positive_rate": old_fpr,
        "new_false_positive_rate": new_fpr,
        "youdens_j": best_j,
    })

    return {
        "success": True,
        "pairs_count": len(pairs),
        "old_threshold": round(old_threshold, 3),
        "old_fpr": round(old_fpr, 3),
        "old_tpr": round(old_tpr, 3),
        "new_threshold": round(new_threshold, 3),
        "new_fpr": round(new_fpr, 3),
        "new_tpr": round(new_tpr, 3),
        "youdens_j": round(best_j, 3),
        "improvement": round(old_fpr - new_fpr, 3),
    }


def generate_scenarios_action(count: int = 10) -> int:
    """Generates synthetic scenario traces in batch."""
    import random
    from scripts.generate_dataset import SCENARIO_POOL
    from agents.support_agent import execute_and_observe

    created = 0
    for _ in range(count):
        _, templates = random.choice(SCENARIO_POOL)
        prompt = random.choice(templates)
        execute_and_observe(prompt, auto_score=True, persist=True)
        created += 1
    return created
"""
Feedback loop: uses accumulated human reviews (confirmed / rejected
classifications) to recalibrate the anomaly-flagging threshold.

Methodology: for every reviewed trace, we have (anomaly_score, confirmed)
where confirmed=True means "this was a genuine issue" and confirmed=False
means "false positive, the auditor/ML were wrong to flag this". We search
over candidate thresholds and pick the one that maximizes Youden's J
statistic (true positive rate - false positive rate) on the reviewed set --
a standard, explainable approach for binary threshold selection (same idea
behind picking an operating point on an ROC curve).

This directly answers "does the system get better with feedback": we
report the false-positive rate at the OLD threshold vs. the NEW threshold
on the same reviewed data, which is the evaluation result referenced in
the project abstract.

Run with:
    python ml/feedback_loop.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.database import SessionLocal, TraceRecord
from ml.threshold_config import load_threshold, save_threshold

MIN_REVIEWS_REQUIRED = 5  # below this, recalibration isn't statistically meaningful


def load_reviewed_traces() -> list[tuple[float, bool]]:
    """Returns (anomaly_score, confirmed) pairs for every trace with a human review."""
    db = SessionLocal()
    try:
        records = db.query(TraceRecord).filter(TraceRecord.anomaly_score.isnot(None)).all()
        pairs = []
        for r in records:
            full_trace = json.loads(r.full_trace_json)
            review = full_trace.get("human_review")
            if review and review.get("confirmed") is not None:
                pairs.append((r.anomaly_score, review["confirmed"]))
        return pairs
    finally:
        db.close()


def false_positive_rate_at_threshold(pairs: list[tuple[float, bool]], threshold: float) -> tuple[float, float]:
    """
    Returns (false_positive_rate, true_positive_rate) at a given threshold,
    evaluated ONLY on the reviewed set (the only place we have ground truth).

    false positive = flagged (score >= threshold) but confirmed=False
    true positive  = flagged (score >= threshold) and confirmed=True
    """
    flagged = [(score, confirmed) for score, confirmed in pairs if score >= threshold]
    not_flagged = [(score, confirmed) for score, confirmed in pairs if score < threshold]

    total_negatives = sum(1 for _, c in pairs if c is False)
    total_positives = sum(1 for _, c in pairs if c is True)

    false_positives = sum(1 for _, c in flagged if c is False)
    true_positives = sum(1 for _, c in flagged if c is True)

    fpr = (false_positives / total_negatives) if total_negatives else 0.0
    tpr = (true_positives / total_positives) if total_positives else 0.0
    return fpr, tpr


def find_best_threshold(pairs: list[tuple[float, bool]]) -> tuple[float, float]:
    """
    Searches candidate thresholds (every distinct score in the reviewed set)
    and returns the one maximizing Youden's J = TPR - FPR, along with that J.
    """
    candidate_thresholds = sorted(set(score for score, _ in pairs))
    best_threshold = candidate_thresholds[0]
    best_j = -1.0

    for t in candidate_thresholds:
        fpr, tpr = false_positive_rate_at_threshold(pairs, t)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = t

    return best_threshold, best_j


def run_feedback_loop() -> None:
    pairs = load_reviewed_traces()

    print(f"Found {len(pairs)} reviewed traces with ground truth.")
    if len(pairs) < MIN_REVIEWS_REQUIRED:
        print(f"⚠️  Need at least {MIN_REVIEWS_REQUIRED} human reviews to recalibrate "
              f"meaningfully. Review more traces in the dashboard first.")
        return

    old_threshold = load_threshold()
    old_fpr, old_tpr = false_positive_rate_at_threshold(pairs, old_threshold)

    new_threshold, best_j = find_best_threshold(pairs)
    new_fpr, new_tpr = false_positive_rate_at_threshold(pairs, new_threshold)

    print(f"\n=== Recalibration Report ===")
    print(f"Reviewed traces used: {len(pairs)} "
          f"({sum(1 for _, c in pairs if c)} confirmed real, "
          f"{sum(1 for _, c in pairs if not c)} confirmed false-positive)")
    print(f"\nOLD threshold: {old_threshold:.3f}")
    print(f"  false_positive_rate: {old_fpr:.1%}   true_positive_rate (recall): {old_tpr:.1%}")
    print(f"\nNEW threshold: {new_threshold:.3f}  (Youden's J = {best_j:.3f})")
    print(f"  false_positive_rate: {new_fpr:.1%}   true_positive_rate (recall): {new_tpr:.1%}")

    improvement = old_fpr - new_fpr
    print(f"\nFalse-positive rate change: {improvement:+.1%} "
          f"({'improved' if improvement > 0 else 'no improvement yet -- more reviews needed' if improvement == 0 else 'worse, check for label noise'})")

    save_threshold(new_threshold, metadata={
        "reviewed_traces_used": len(pairs),
        "old_threshold": old_threshold,
        "old_false_positive_rate": old_fpr,
        "new_false_positive_rate": new_fpr,
        "youdens_j": best_j,
    })
    print(f"\n✅ New threshold ({new_threshold:.3f}) saved to ml/threshold_config.json")
    print("   auditor/run_auditor.py and the dashboard will now use this updated threshold.")


if __name__ == "__main__":
    run_feedback_loop()
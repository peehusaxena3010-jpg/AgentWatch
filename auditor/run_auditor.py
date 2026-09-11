"""
Runs the LLM auditor over all traces flagged as anomalous by the ML model,
and writes each classification (failure_mode + explanation) back into the
trace's auditor_classification field in the database.

A trace is considered "flagged" here if its anomaly_score (written by
ml/train_anomaly_model.py) is at or above FLAG_THRESHOLD. This threshold is
a simple stand-in for the ML model's binary prediction -- feel free to
tune it once you have a feel for your own score distribution.

Run with:
    python -m auditor.run_auditor
(must run as a module, same reason as auditor.py itself -- see its comments)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.database import SessionLocal, TraceRecord  # noqa: E402
from auditor.auditor import classify_trace, MOCK_MODE  # noqa: E402
from ml.threshold_config import load_threshold  # noqa: E402

FLAG_THRESHOLD = load_threshold()
DELAY_BETWEEN_CALLS_SECONDS = 0.05 if MOCK_MODE else 4


def run_audit() -> None:
    db = SessionLocal()
    try:
        flagged_records = (
            db.query(TraceRecord)
            .filter(TraceRecord.anomaly_score >= FLAG_THRESHOLD)
            .all()
        )

        if not flagged_records:
            print(f"No traces found with anomaly_score >= {FLAG_THRESHOLD}. "
                  f"Run ml/train_anomaly_model.py first if you haven't.")
            return

        print(f"Auditing {len(flagged_records)} flagged traces...\n")

        succeeded = 0
        failed = 0

        for record in flagged_records:
            full_trace = json.loads(record.full_trace_json)

            try:
                classification = classify_trace(full_trace)
            except Exception as exc:  # noqa: BLE001
                # Don't let one bad call (rate limit, timeout, etc.) kill the
                # whole batch and lose every classification done so far.
                print(f"--- Trace {record.trace_id} (anomaly_score={record.anomaly_score:.2f}) ---")
                print(f"  ⚠️  classification failed: {exc}")
                print(f"  (skipped -- rerun the script later to retry this trace)\n")
                failed += 1
                continue

            full_trace["auditor_classification"] = classification
            record.full_trace_json = json.dumps(full_trace)

            # Commit immediately after each trace, not just once at the end --
            # so a later failure can never wipe out earlier successful work.
            db.commit()
            succeeded += 1

            print(f"--- Trace {record.trace_id} (anomaly_score={record.anomaly_score:.2f}) ---")
            print(f"  failure_mode: {classification.get('failure_mode')}")
            print(f"  explanation: {classification.get('explanation')}")
            print()

            time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        print(f"✅ Done. {succeeded} traces classified and saved, {failed} skipped due to errors.")
        if failed:
            print("   Rerun this script to retry the skipped ones (already-classified traces are unaffected).")

    finally:
        db.close()


if __name__ == "__main__":
    run_audit()
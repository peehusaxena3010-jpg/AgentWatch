"""
Batch runner: Evaluates response faithfulness across all traces in the database.

Treats the tool evidence history as the premise and the final response as the
hypothesis, calculating a faithfulness score [0.0 - 1.0] for every trace.

Run with:
    python -m ml.run_faithfulness_scorer
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ml.faithfulness_scorer import FaithfulnessScorer


def run():
    print("Running Faithfulness Scorer across all stored traces...")
    count = FaithfulnessScorer.score_and_update_db()
    print(f"✅ Successfully evaluated and updated faithfulness scores for {count} traces.")


if __name__ == "__main__":
    run()

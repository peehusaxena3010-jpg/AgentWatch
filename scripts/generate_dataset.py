"""
Batch trace generator — larger, randomized scenario pool.

Generates realistic agent traces across customer support scenarios:
- Valid refunds for delayed delivery (Order #4521)
- Standard order delivery inquiries (Order #1001)
- In-flight order processing queries (Order #2002)
- Double refund attempts / Policy violations (Order #3003)
- Non-existent orders / Wrong tool calls (Order #9999)
- Repeated tool calls / Infinite loop triggers (Loop scenario)
- Ambiguous queries triggering human escalation

Stores via API if running, or directly to SQLite database.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
from agents.support_agent import run_agent, API_BASE_URL
from backend.database import SessionLocal, TraceRecord, init_db

SCENARIO_POOL = [
    ("normal_refund_valid", [
        "I want a refund for order #4521, it never arrived.",
        "Refund order #4521 please, it's very late.",
        "Order #4521 hasn't shown up, can I get a refund?",
    ]),
    ("normal_status_check", [
        "Can you tell me the status of order #1001?",
        "What's happening with order #1001?",
        "Where is my order #1001?",
    ]),
    ("normal_delivered_no_issue", [
        "When will order #2002 arrive?",
        "Status update on order #2002 please.",
    ]),
    ("edge_already_refunded", [
        "I'd like a refund for order #3003, it's broken.",
        "Order #3003 arrived damaged, refund please.",
    ]),
    ("edge_unknown_order", [
        "Please refund order #9999, I never got it.",
        "Refund order #9999, it's missing.",
    ]),
    ("edge_ambiguous_request", [
        "My order is messed up, fix it please.",
        "Something's wrong with my order, help.",
    ]),
    ("edge_loop_trigger", [
        "Can you double check order #4521, loop through it carefully?",
        "Please loop-verify order #4521 status thoroughly.",
    ]),
    ("normal_refund_valid_repeat", [
        "Refund order #4521, it's late.",
    ]),
]

SAMPLE_SIZE = 25


def check_api_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False


def main(sample_size: int = SAMPLE_SIZE):
    init_db()
    random.seed()

    api_active = check_api_alive()
    mode_str = f"REST API ({API_BASE_URL})" if api_active else "Direct SQLite Storage"
    print(f"Target Storage: {mode_str}")
    print(f"Generating and executing {sample_size} varied support scenarios...\n")

    stored_count = 0
    db = SessionLocal() if not api_active else None

    try:
        for i in range(1, sample_size + 1):
            label, templates = random.choice(SCENARIO_POOL)
            user_input = random.choice(templates)
            print(f"[{i:02d}/{sample_size}] [{label}] \"{user_input}\"")
            trace = run_agent(user_input)

            if api_active:
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/traces",
                        data=trace.model_dump_json(),
                        headers={"Content-Type": "application/json"},
                        timeout=2,
                    )
                    success = res.status_code in (200, 201)
                except Exception:
                    success = False
            else:
                record = TraceRecord(
                    trace_id=str(trace.trace_id),
                    session_id=str(trace.session_id),
                    agent_name=trace.agent_name,
                    agent_version=trace.agent_version,
                    timestamp=trace.timestamp,
                    status=trace.status,
                    anomaly_score=trace.anomaly_score,
                    faithfulness_score=trace.faithfulness_score,
                    total_latency_ms=trace.total_latency_ms,
                    full_trace_json=trace.model_dump_json(),
                )
                db.add(record)
                db.commit()
                success = True

            if success:
                stored_count += 1
            print(f"       -> status={trace.status}, steps={len(trace.steps)}, latency={trace.total_latency_ms}ms (saved: {success})")

        print(f"\n✅ Done. {stored_count}/{sample_size} traces generated and stored successfully.")
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_SIZE
    main(count)
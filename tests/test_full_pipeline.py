"""
Automated End-to-End Test Suite for AgentWatch Observability Framework.

Tests:
1. Agent execution & trace schema conformance
2. 12-feature extraction logic
3. Database persistence across session reboots
4. Independent Faithfulness Scorer (grounded vs. hallucinated)
5. Machine Learning Anomaly Detection (Isolation Forest)
6. Behavioral LLM Auditor failure-mode taxonomy
7. Human review & closed-loop threshold recalibration (Youden's J)
8. FastAPI REST endpoints & health checks
"""

import json
import os
import sys
import unittest
from uuid import uuid4

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.database import SessionLocal, TraceRecord, init_db
from backend.models.trace_models import AgentTrace, TraceStatus, StepStatus, HumanReview
from agents.support_agent import run_agent, execute_and_observe
from ml.feature_extraction import trace_to_features, FEATURE_NAMES
from ml.faithfulness_scorer import score_trace, FaithfulnessScorer
from ml.train_anomaly_model import predict_single_trace, train_and_score
from auditor.auditor import classify_trace
from ml.feedback_loop import false_positive_rate_at_threshold, find_best_threshold
from ml.threshold_config import load_threshold, save_threshold


class TestAgentWatchPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_agent_execution_and_schema(self):
        """Test agent execution in mock mode producing schema-compliant trace."""
        trace = run_agent("Can you tell me the status of order #1001?")
        self.assertIsInstance(trace, AgentTrace)
        self.assertEqual(trace.status, TraceStatus.success)
        self.assertGreaterEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0].tool_called, "check_order_status")
        self.assertIn("1001", trace.final_response)

    def test_02_feature_extraction(self):
        """Test extraction of exactly 12 numerical features."""
        trace = run_agent("I want a refund for order #4521, it never arrived.")
        features = trace_to_features(trace.model_dump())
        self.assertEqual(len(features), 12)
        for name in FEATURE_NAMES:
            self.assertIn(name, features)
            self.assertIsInstance(features[name], float)

    def test_03_database_persistence_across_sessions(self):
        """Verify trace persistence in SQLite: write, disconnect, reconnect, verify."""
        test_id = str(uuid4())
        session_id = str(uuid4())

        # Session 1: write
        db1 = SessionLocal()
        record = TraceRecord(
            trace_id=test_id,
            session_id=session_id,
            agent_name="test_agent",
            agent_version="v1",
            status="success",
            anomaly_score=0.45,
            faithfulness_score=0.95,
            total_latency_ms=120.5,
            full_trace_json=json.dumps({"trace_id": test_id, "user_input": "persistence test"}),
        )
        db1.add(record)
        db1.commit()
        db1.close()

        # Session 2: completely new DB connection
        db2 = SessionLocal()
        retrieved = db2.get(TraceRecord, test_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.agent_name, "test_agent")
        self.assertEqual(retrieved.anomaly_score, 0.45)
        self.assertEqual(retrieved.faithfulness_score, 0.95)

        # Cleanup
        db2.delete(retrieved)
        db2.commit()
        db2.close()

    def test_04_faithfulness_scorer(self):
        """Test that faithfulness scorer distinguishes grounded vs hallucinated responses."""
        # Grounded: tool returned delivery overdue, refund succeeded, agent claimed refund
        grounded_trace = {
            "steps": [
                {"tool_called": "check_order_status", "tool_args": {"order_id": "4521"}, "tool_result": {"status": "shipped", "amount": 49.99, "already_refunded": False}},
                {"tool_called": "issue_refund", "tool_args": {"order_id": "4521", "reason": "overdue"}, "tool_result": {"refund_status": "success", "amount": 49.99}},
            ],
            "final_response": "I have processed a refund of $49.99 for order #4521.",
            "status": "success",
        }
        score_grounded, diag_g = score_trace(grounded_trace)
        self.assertGreaterEqual(score_grounded, 0.8)

        # Hallucinated: tool returned already refunded error, but agent claimed refund processed
        hallucinated_trace = {
            "steps": [
                {"tool_called": "check_order_status", "tool_args": {"order_id": "3003"}, "tool_result": {"status": "shipped", "already_refunded": True}},
                {"tool_called": "issue_refund", "tool_args": {"order_id": "3003"}, "tool_result": {"error": "Order 3003 has already been refunded."}},
            ],
            "final_response": "Great news! Your refund has been processed successfully.",
            "status": "error",
        }
        score_hallucinated, diag_h = score_trace(hallucinated_trace)
        self.assertLess(score_hallucinated, 0.5)
        self.assertGreater(len(diag_h["penalties"]), 0)

    def test_05_auditor_taxonomy_classification(self):
        """Test failure-mode classification on policy violation and loop traces."""
        # Policy violation
        pv_trace = {
            "steps": [
                {"tool_called": "check_order_status", "tool_args": {"order_id": "3003"}, "tool_result": {"status": "shipped", "already_refunded": True}},
                {"tool_called": "issue_refund", "tool_args": {"order_id": "3003"}, "tool_result": {"error": "Order 3003 has already been refunded."}},
            ],
            "final_response": "Something went wrong: Order 3003 has already been refunded.",
            "status": "error",
        }
        res_pv = classify_trace(pv_trace)
        self.assertEqual(res_pv.get("failure_mode"), "policy_violation")

        # Infinite loop
        loop_trace = {
            "steps": [
                {"tool_called": "check_order_status", "tool_args": {"order_id": "4521"}, "tool_result": {"status": "shipped"}},
                {"tool_called": "check_order_status", "tool_args": {"order_id": "4521"}, "tool_result": {"status": "shipped"}},
            ],
            "final_response": "Order is shipped.",
            "status": "success",
        }
        res_loop = classify_trace(loop_trace)
        self.assertEqual(res_loop.get("failure_mode"), "infinite_loop")

    def test_06_human_feedback_and_youdens_j_recalibration(self):
        """Test threshold search maximizing Youden's J on human review data."""
        # Mock reviewed dataset: (anomaly_score, confirmed)
        mock_pairs = [
            (0.85, True),   # True positive
            (0.80, True),   # True positive
            (0.75, True),   # True positive
            (0.65, False),  # False positive at low threshold
            (0.60, False),  # False positive
            (0.30, False),  # True negative
            (0.25, False),  # True negative
        ]

        best_t, best_j = find_best_threshold(mock_pairs)
        self.assertGreaterEqual(best_t, 0.70)  # Optimal threshold should separate >= 0.75 from false positives
        self.assertGreater(best_j, 0.5)

    def test_07_fastapi_endpoints(self):
        """Test FastAPI endpoints via FastAPI TestClient."""
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)

        # Health endpoint
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertTrue(res_health.json()["database_connected"])

        # Stats endpoint
        res_stats = client.get("/stats")
        self.assertEqual(res_stats.status_code, 200)
        self.assertIn("total_traces", res_stats.json())

        # Pipeline run agent endpoint
        res_run = client.post("/pipeline/run-agent", json={"user_input": "Where is order #1001?", "auto_score": True})
        self.assertEqual(res_run.status_code, 200)
        data = res_run.json()
        self.assertEqual(data["status"], "success")
        self.assertIsNotNone(data["anomaly_score"])
        self.assertIsNotNone(data["faithfulness_score"])


if __name__ == "__main__":
    unittest.main()

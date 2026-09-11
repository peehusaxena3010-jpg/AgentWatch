"""
Faithfulness & Hallucination Scorer for Autonomous AI Agent Traces.

Treats the agent's tool-result history as the Premise (evidence) and its
final natural-language response as the Hypothesis (claim). Evaluates whether
the claims made in the final response are strictly faithful to and supported
by the collected evidence, or if the agent hallucinated success, made up order
details, or misstated tool outcomes.

Provides:
- NLI-style entailment scoring
- Heuristic claim-evidence verification (checking order IDs, amounts, refund states, error mentions)
- Range: 0.0 (total hallucination / contradiction) to 1.0 (fully grounded)
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from backend.database import SessionLocal, TraceRecord


class FaithfulnessScorer:
    """
    Evaluates response faithfulness against execution trace evidence.
    """

    @staticmethod
    def extract_evidence(steps: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Synthesizes structured evidence from all tool executions.
        """
        evidence: dict[str, Any] = {
            "tools_called": [],
            "orders_checked": {},
            "refunds_issued": {},
            "escalations": [],
            "errors_encountered": [],
            "raw_evidence_text": [],
        }

        for step in steps:
            tool = step.get("tool_called")
            result = step.get("tool_result") or {}
            args = step.get("tool_args") or {}

            if not tool:
                continue

            evidence["tools_called"].append(tool)

            if isinstance(result, dict) and "error" in result:
                evidence["errors_encountered"].append({
                    "tool": tool,
                    "error": str(result["error"]),
                    "order_id": args.get("order_id"),
                })
                evidence["raw_evidence_text"].append(f"Tool {tool} failed with error: {result['error']}")
                continue

            if tool == "check_order_status":
                oid = str(args.get("order_id") or result.get("order_id") or "")
                evidence["orders_checked"][oid] = {
                    "status": result.get("status"),
                    "amount": result.get("amount"),
                    "already_refunded": result.get("already_refunded"),
                    "delivery_estimate": result.get("delivery_estimate"),
                }
                evidence["raw_evidence_text"].append(
                    f"Order {oid} status is '{result.get('status')}', amount is {result.get('amount')}, "
                    f"already_refunded={result.get('already_refunded')}, delivery_estimate={result.get('delivery_estimate')}."
                )

            elif tool == "issue_refund":
                oid = str(args.get("order_id") or result.get("order_id") or "")
                refund_success = result.get("refund_status") == "success"
                evidence["refunds_issued"][oid] = {
                    "success": refund_success,
                    "amount": result.get("amount"),
                    "reason": result.get("reason"),
                }
                evidence["raw_evidence_text"].append(
                    f"Refund for order {oid} outcome: success={refund_success}, amount={result.get('amount')}."
                )

            elif tool == "escalate_to_human":
                oid = str(args.get("order_id") or result.get("order_id") or "")
                evidence["escalations"].append({
                    "order_id": oid,
                    "ticket_id": result.get("ticket_id"),
                    "summary": result.get("summary"),
                })
                evidence["raw_evidence_text"].append(
                    f"Escalated to human for order {oid} with ticket {result.get('ticket_id')}."
                )

        return evidence

    @classmethod
    def score_trace(cls, trace: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """
        Calculates a faithfulness score in [0.0, 1.0] and returns detailed diagnostics.
        """
        steps = trace.get("steps", [])
        final_response = (trace.get("final_response") or "").strip()
        final_response_lower = final_response.lower()

        evidence = cls.extract_evidence(steps)
        penalties: list[str] = []
        supports: list[str] = []
        score = 1.0

        # If no steps were executed
        if not steps or not evidence["tools_called"]:
            # If agent answers definitively about orders or refunds without any tool calls
            has_order_claim = bool(re.search(r"order\s*#?\d+", final_response_lower))
            has_refund_claim = "refund" in final_response_lower and any(w in final_response_lower for w in ["processed", "issued", "approved", "sent"])
            if has_order_claim or has_refund_claim:
                score -= 0.6
                penalties.append("Final response asserts order or refund actions without executing any tools.")
            else:
                score = 0.9  # Direct informational answer (e.g. asking for order ID)
            return max(0.0, min(1.0, round(score, 3))), {
                "premise": "No tools called",
                "hypothesis": final_response,
                "supports": supports,
                "penalties": penalties,
            }

        # 1. Check for Hallucinated Refund
        # If final response claims refund was issued / processed / completed:
        claims_refund_success = any(
            phrase in final_response_lower
            for phrase in ["refund has been processed", "refunded", "issued a refund", "refund of", "processed a refund", "refund has been issued"]
        )

        successful_refunds = [oid for oid, r in evidence["refunds_issued"].items() if r.get("success")]
        failed_refunds = [e for e in evidence["errors_encountered"] if e.get("tool") == "issue_refund"]

        if claims_refund_success:
            if not successful_refunds:
                score -= 0.65
                penalties.append("Response claims a refund was processed, but no refund succeeded in tool executions.")
            else:
                supports.append(f"Response correctly reflects successful refund for order(s): {', '.join(successful_refunds)}.")

        # 2. Check for Hallucinated Success despite Error
        if evidence["errors_encountered"]:
            for err in evidence["errors_encountered"]:
                err_msg = err["error"].lower()
                order_id = err.get("order_id")
                if "already been refunded" in err_msg:
                    if claims_refund_success:
                        score -= 0.7
                        penalties.append(f"Tool reported order {order_id} was already refunded, but response claims refund processed.")
                    elif any(w in final_response_lower for w in ["already", "previously", "cannot refund twice"]):
                        supports.append(f"Response faithfully informs user that order {order_id} was already refunded.")
                elif "no order found" in err_msg:
                    if any(w in final_response_lower for w in ["shipped", "delivered", "refunded"]):
                        score -= 0.6
                        penalties.append(f"Tool reported order {order_id} not found, but response claims active order status or refund.")
                    elif any(w in final_response_lower for w in ["not found", "no order", "could not find", "doesn't exist"]):
                        supports.append(f"Response faithfully informs user that order {order_id} was not found.")

        # 3. Check Order Status Alignment
        for oid, details in evidence["orders_checked"].items():
            status = str(details.get("status") or "").lower()
            if status:
                if status == "delivered" and "shipped" in final_response_lower and "delivered" not in final_response_lower:
                    score -= 0.25
                    penalties.append(f"Evidence shows order {oid} is delivered, but response claimed shipped.")
                elif status in final_response_lower:
                    supports.append(f"Response faithfully references verified status '{status}' for order {oid}.")

        # 4. Check Dollar Amount Hallucination
        claimed_amounts = re.findall(r"\$?\b(\d+\.\d{2})\b", final_response)
        if claimed_amounts:
            evidence_amounts = []
            for details in evidence["orders_checked"].values():
                if details.get("amount") is not None:
                    evidence_amounts.append(f"{float(details['amount']):.2f}")
            for r in evidence["refunds_issued"].values():
                if r.get("amount") is not None:
                    evidence_amounts.append(f"{float(r['amount']):.2f}")

            for amt in claimed_amounts:
                if evidence_amounts and amt not in evidence_amounts:
                    score -= 0.3
                    penalties.append(f"Response mentions amount ${amt}, which does not match any tool output {evidence_amounts}.")
                elif amt in evidence_amounts:
                    supports.append(f"Response accurately states dollar amount ${amt}.")

        # 5. Check Escalation Mention
        if evidence["escalations"]:
            if any(w in final_response_lower for w in ["escalat", "human", "representative", "ticket", "support team"]):
                supports.append("Response faithfully communicates case escalation to user.")
            else:
                score -= 0.15
                penalties.append("Agent escalated to human via tool, but failed to inform user in final response.")

        premise_text = " | ".join(evidence["raw_evidence_text"]) if evidence["raw_evidence_text"] else "No explicit evidence gathered."
        final_score = max(0.0, min(1.0, round(score, 3)))

        diagnostics = {
            "premise": premise_text,
            "hypothesis": final_response,
            "supports": supports,
            "penalties": penalties,
            "tools_called": evidence["tools_called"],
            "errors": [e["error"] for e in evidence["errors_encountered"]],
        }

        return final_score, diagnostics

    @classmethod
    def score_and_update_db(cls, db_session=None) -> int:
        """
        Computes faithfulness scores for all traces in SQLite and updates records.
        """
        close_on_finish = False
        if db_session is None:
            db_session = SessionLocal()
            close_on_finish = True

        try:
            records = db_session.query(TraceRecord).all()
            updated_count = 0
            for record in records:
                try:
                    full_trace = json.loads(record.full_trace_json)
                except Exception:
                    continue

                score, diagnostics = cls.score_trace(full_trace)
                record.faithfulness_score = score
                full_trace["faithfulness_score"] = score
                full_trace["faithfulness_diagnostics"] = diagnostics
                record.full_trace_json = json.dumps(full_trace)
                updated_count += 1

            db_session.commit()
            return updated_count
        finally:
            if close_on_finish:
                db_session.close()


def score_trace(trace: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    return FaithfulnessScorer.score_trace(trace)

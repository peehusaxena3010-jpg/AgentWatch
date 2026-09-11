"""
LLM-based auditor: reads a full trace and classifies WHY it looks wrong,
producing a structured failure_mode + human-readable explanation.

This is deliberately a separate model call from the agent itself -- the
auditor is reviewing the agent's work after the fact, with the benefit of
seeing the whole trace at once (all steps, all tool results, the final
response), rather than reasoning step-by-step like the agent did.

MOCK_MODE:
    True  -> high-fidelity heuristic classification, no API calls.
    False -> real Gemini 2.5 Flash call via google-genai, returning structured JSON.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from auditor.taxonomy import FAILURE_MODE_DESCRIPTIONS, format_taxonomy_for_prompt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Determine mock mode: True if explicitly configured or if GEMINI_API_KEY is missing
_API_KEY = os.environ.get("GEMINI_API_KEY")
_EXPLICIT_MOCK = os.environ.get("MOCK_MODE")
if _EXPLICIT_MOCK is not None:
    MOCK_MODE = _EXPLICIT_MOCK.lower() in ("true", "1", "yes")
else:
    MOCK_MODE = not bool(_API_KEY and _API_KEY.strip() and _API_KEY != "your_key_here")


AUDITOR_SYSTEM_PROMPT = f"""You are an AI agent auditor. You will be given the
full execution trace of an autonomous support agent -- its reasoning at each
step, which tools it called, what those tools returned, and its final
response to the user.

Your job is to determine whether the agent's behavior falls into one of the
following failure categories, or whether it behaved correctly ("none"):

{format_taxonomy_for_prompt()}

Respond with ONLY a JSON object in this exact format, with no other text:
{{"failure_mode": "<one of the category names above>", "explanation": "<one or two sentences explaining your reasoning, referencing specific steps/tool results>"}}
"""


def _format_trace_for_prompt(trace: dict) -> str:
    """Renders a trace's steps and outcome as readable text for the auditor prompt."""
    lines = [f"User request: {trace.get('user_input')}\n"]
    for step in trace.get("steps", []):
        lines.append(f"Step {step.get('step_number')}:")
        lines.append(f"  Reasoning: {step.get('decision_rationale')}")
        if step.get("tool_called"):
            lines.append(f"  Tool called: {step['tool_called']}({step.get('tool_args')})")
            lines.append(f"  Tool result: {step.get('tool_result')}")
        lines.append(f"  Step status: {step.get('step_status')}")
    lines.append(f"\nFinal response to user: {trace.get('final_response')}")
    lines.append(f"Overall trace status: {trace.get('status')}")
    return "\n".join(lines)


def classify_trace_mock(trace: dict) -> dict:
    """
    Comprehensive rule-based heuristic classification covering the 8 failure modes.
    Used for local testing and reliable zero-cost demonstrations.
    """
    steps = trace.get("steps", [])
    final_resp = (trace.get("final_response") or "").lower()

    # 1. Infrastructure error
    if trace.get("status") == "error" and trace.get("error_message"):
        return {
            "failure_mode": "infrastructure_error",
            "explanation": f"The agent's execution was interrupted by an infrastructure-level failure: {trace.get('error_message')[:200]}",
        }

    # 2. Infinite loop detection
    tool_sequence = [s.get("tool_called") for s in steps if s.get("tool_called")]
    for i in range(1, len(tool_sequence)):
        if tool_sequence[i] == tool_sequence[i - 1]:
            return {
                "failure_mode": "infinite_loop",
                "explanation": f"The agent called '{tool_sequence[i]}' repeatedly in consecutive steps without making forward progress.",
            }

    # 3. Check for Hallucinated Success
    # If a tool failed with an error, but the final response claims refund was issued or success
    has_error_tool = any(
        isinstance(s.get("tool_result"), dict) and "error" in s.get("tool_result", {})
        for s in steps
    )
    claims_success = any(
        w in final_resp for w in ["refund has been processed", "processed a refund", "refunded successfully", "handled successfully"]
    )
    if has_error_tool and claims_success and trace.get("status") != "error":
        return {
            "failure_mode": "hallucinated_success",
            "explanation": "The agent claimed the request or refund succeeded even though underlying tool calls failed with an error.",
        }

    # 4. Unauthorized action / Policy violation
    # Check if refund was issued without checking order status first
    tool_calls_in_order = [s.get("tool_called") for s in steps if s.get("tool_called")]
    if "issue_refund" in tool_calls_in_order:
        refund_idx = tool_calls_in_order.index("issue_refund")
        if "check_order_status" not in tool_calls_in_order[:refund_idx]:
            return {
                "failure_mode": "unauthorized_action",
                "explanation": "The agent attempted to issue a refund without first verifying order status via check_order_status.",
            }

    # 5. Policy violation vs. wrong tool call from tool results
    for step in steps:
        result = step.get("tool_result")
        if isinstance(result, dict) and "error" in result:
            error_text = str(result["error"]).lower()
            if "already been refunded" in error_text:
                return {
                    "failure_mode": "policy_violation",
                    "explanation": f"The agent attempted to refund an order that had already been refunded, violating the no-double-refund policy. Result: {result['error']}",
                }
            if "no order found" in error_text:
                return {
                    "failure_mode": "wrong_tool_call",
                    "explanation": f"The agent called a tool with an invalid or non-existent order ID: {result['error']}",
                }

    # 6. Low confidence answered anyway
    if trace.get("status") == "incomplete":
        return {
            "failure_mode": "low_confidence_answered_anyway",
            "explanation": "The agent did not reach a final response within allowed steps or answered definitively despite ambiguity.",
        }

    return {
        "failure_mode": "none",
        "explanation": "No issues detected -- tool usage and final response are consistent with available evidence.",
    }


def classify_trace_real(trace: dict) -> dict:
    """Real Gemini 2.5 Flash call, asked to return structured JSON."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        return classify_trace_mock(trace)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
        trace_text = _format_trace_for_prompt(trace)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=trace_text)])],
            config=types.GenerateContentConfig(
                system_instruction=AUDITOR_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        parsed = json.loads(response.text)
        if parsed.get("failure_mode") not in FAILURE_MODE_DESCRIPTIONS:
            parsed["failure_mode"] = "none"
        return parsed
    except Exception as exc:
        print(f"[auditor] Gemini call failed ({exc}), falling back to heuristic mock audit.")
        mock_res = classify_trace_mock(trace)
        mock_res["explanation"] += f" (Note: Gemini API fallback due to: {exc})"
        return mock_res


def classify_trace(trace: dict) -> dict:
    """Entry point: dispatches to mock or real classification based on configuration."""
    if MOCK_MODE:
        return classify_trace_mock(trace)
    return classify_trace_real(trace)


if __name__ == "__main__":
    example_trace = {
        "user_input": "Refund order #3003, it's broken.",
        "steps": [
            {
                "step_number": 1,
                "decision_rationale": "Checking order status before refunding.",
                "tool_called": "check_order_status",
                "tool_args": {"order_id": "3003"},
                "tool_result": {"status": "shipped", "already_refunded": True},
                "step_status": "success",
            },
            {
                "step_number": 2,
                "decision_rationale": "Attempting refund.",
                "tool_called": "issue_refund",
                "tool_args": {"order_id": "3003", "reason": "damaged"},
                "tool_result": {"error": "Order 3003 has already been refunded."},
                "step_status": "failure",
            },
        ],
        "final_response": "Something went wrong: Order 3003 has already been refunded.",
        "status": "error",
    }
    result = classify_trace(example_trace)
    print(json.dumps(result, indent=2))
    print("\n✅ Auditor classification works.")
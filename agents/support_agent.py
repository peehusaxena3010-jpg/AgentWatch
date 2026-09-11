"""
Support agent: tool-calling loop using Gemini 2.5 Flash or scenario-aware mock.

Every reasoning step, tool call, and result is logged into the AgentTrace
schema (backend/models/trace_models.py) as it happens.

MOCK_MODE:
    True  -> no API calls made. Realistic scenario-aware responses are generated.
    False -> real calls to Gemini 2.5 Flash via google-genai SDK.
             Requires GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from uuid import uuid4
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.models.trace_models import (
    AgentTrace,
    Step,
    StepStatus,
    TraceStatus,
    AuditorClassification,
)
from agents.tools import TOOL_REGISTRY, TOOL_DECLARATIONS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_API_KEY = os.environ.get("GEMINI_API_KEY")
_EXPLICIT_MOCK = os.environ.get("MOCK_MODE")
if _EXPLICIT_MOCK is not None:
    MOCK_MODE = _EXPLICIT_MOCK.lower() in ("true", "1", "yes")
else:
    MOCK_MODE = not bool(_API_KEY and _API_KEY.strip() and _API_KEY != "your_key_here")

MAX_STEPS = 6
AGENT_NAME = "support_agent"
AGENT_VERSION = "v1"

SYSTEM_PROMPT = """You are a customer support agent. You have access to tools to
check order status, issue refunds, and escalate to a human when needed.
Always check order status before issuing a refund. Never issue a refund twice
for the same order. If you are unsure or the situation is ambiguous, escalate
to a human rather than guessing. Briefly state your reasoning in plain text
before calling a tool."""


# --------------------------------------------------------------------------
# MOCK PATH — scenario-aware scripted responses
# --------------------------------------------------------------------------

def scripted_plan(user_input: str):
    """
    Returns a list of (tool_name, tool_args, reasoning) steps based on what
    the input actually contains.
    """
    ui = user_input.lower()

    if "4521" in ui and "loop" not in ui:
        return [
            ("check_order_status", {"order_id": "4521"}, "Checking order status before refunding."),
            ("issue_refund", {"order_id": "4521", "reason": "not_delivered"}, "Delivery overdue, issuing refund."),
        ]
    if "9999" in ui:
        return [
            ("check_order_status", {"order_id": "9999"}, "Checking order status."),
        ]  # order doesn't exist -> tool returns error -> failure trace
    if "3003" in ui:
        return [
            ("check_order_status", {"order_id": "3003"}, "Checking order status before refunding."),
            ("issue_refund", {"order_id": "3003", "reason": "damaged"}, "Attempting refund."),
        ]  # already refunded -> tool returns error -> failure trace
    if "1001" in ui:
        return [
            ("check_order_status", {"order_id": "1001"}, "Checking order status."),
        ]
    if "2002" in ui:
        return [
            ("check_order_status", {"order_id": "2002"}, "Checking order status."),
        ]
    if "loop" in ui:
        return [
            ("check_order_status", {"order_id": "4521"}, "Checking order status."),
            ("check_order_status", {"order_id": "4521"}, "Checking again to be sure."),
            ("check_order_status", {"order_id": "4521"}, "Checking once more."),
        ]  # repeated identical tool call -> loop signal
    # ambiguous / no order ID given
    return [
        ("escalate_to_human", {"order_id": "0000", "issue_summary": "Ambiguous request, no order ID provided."}, "Request is ambiguous, escalating to a human."),
    ]


def run_agent_mock(user_input: str) -> tuple[list[Step], str, TraceStatus, str | None]:
    steps: list[Step] = []
    final_response = ""
    status = TraceStatus.success
    error_message = None

    plan = scripted_plan(user_input)
    inject_latency_spike = random.random() < 0.15

    for i, (tool_name, tool_args, reasoning) in enumerate(plan, start=1):
        step_start = time.perf_counter()
        if inject_latency_spike and i == 1:
            time.sleep(0.2)

        tool_fn = TOOL_REGISTRY.get(tool_name)
        tool_result = tool_fn(**tool_args) if tool_fn else {"error": f"Unknown tool '{tool_name}'"}
        step_status = StepStatus.failure if "error" in tool_result else StepStatus.success

        steps.append(Step(
            step_number=i,
            decision_rationale=reasoning,
            tool_called=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            step_status=step_status,
            latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
        ))

        if step_status == StepStatus.failure:
            status = TraceStatus.error
            final_response = f"Something went wrong: {tool_result.get('error')}"
            break
    else:
        # Build contextual response
        ui = user_input.lower()
        if "4521" in ui:
            final_response = "I have verified that order #4521 was overdue and issued a full refund of $49.99 to your account."
        elif "1001" in ui:
            final_response = "Order #1001 was successfully delivered on 2026-07-15. Amount was $19.99."
        elif "2002" in ui:
            final_response = "Order #2002 is currently processing with an estimated delivery date of 2026-07-25."
        elif "loop" in ui:
            final_response = "I verified order #4521 multiple times. Status is shipped."
        else:
            final_response = "Your request has been escalated to our human support team with a ticket created."
        status = TraceStatus.success

    return steps, final_response, status, error_message


# --------------------------------------------------------------------------
# REAL PATH — Gemini 2.5 Flash multi-turn tool calling
# --------------------------------------------------------------------------

def run_agent_real(user_input: str) -> tuple[list[Step], str, TraceStatus, str | None]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return run_agent_mock(user_input)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        return run_agent_mock(user_input)

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
    tools = [types.Tool(function_declarations=TOOL_DECLARATIONS)]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=tools)

    contents: list = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    steps: list[Step] = []
    final_response = ""
    status = TraceStatus.incomplete
    error_message = None

    for step_number in range(1, MAX_STEPS + 1):
        step_start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=contents, config=config,
            )
        except Exception as exc:
            error_message = str(exc)
            status = TraceStatus.error
            break

        candidate = response.candidates[0]
        contents.append(candidate.content)

        reasoning_text = ""
        function_call = None
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                reasoning_text += part.text
            if getattr(part, "function_call", None):
                function_call = part.function_call

        if function_call is None:
            final_response = reasoning_text or "(model returned an empty response)"
            steps.append(Step(
                step_number=step_number,
                decision_rationale=reasoning_text or "(model produced a direct text response)",
                tool_called=None, tool_args=None, tool_result=None,
                step_status=StepStatus.success,
                latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
            ))
            status = TraceStatus.success
            break

        tool_name = function_call.name
        tool_args = dict(function_call.args)
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            tool_result = {"error": f"Unknown tool '{tool_name}' requested by model."}
            step_status = StepStatus.failure
        else:
            try:
                tool_result = tool_fn(**tool_args)
                step_status = StepStatus.failure if "error" in tool_result else StepStatus.success
            except Exception as exc:
                tool_result = {"error": str(exc)}
                step_status = StepStatus.failure

        steps.append(Step(
            step_number=step_number,
            decision_rationale=reasoning_text or "(model provided no explicit reasoning text)",
            tool_called=tool_name, tool_args=tool_args, tool_result=tool_result,
            step_status=step_status,
            latency_ms=round((time.perf_counter() - step_start) * 1000, 2),
        ))

        function_response_part = types.Part.from_function_response(
            name=tool_name, response=tool_result,
        )
        contents.append(types.Content(role="tool", parts=[function_response_part]))
    else:
        status = TraceStatus.incomplete
        final_response = "(agent did not produce a final response within max steps)"

    return steps, final_response, status, error_message


def run_agent(user_input: str) -> AgentTrace:
    """Runs the tool-calling loop end-to-end and returns a populated AgentTrace."""
    start_time = time.perf_counter()

    if MOCK_MODE:
        steps, final_response, status, error_message = run_agent_mock(user_input)
    else:
        steps, final_response, status, error_message = run_agent_real(user_input)

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return AgentTrace(
        session_id=uuid4(),
        agent_name=AGENT_NAME,
        agent_version=AGENT_VERSION,
        user_input=user_input,
        steps=steps,
        final_response=final_response,
        status=status,
        total_latency_ms=total_latency_ms,
        error_message=error_message,
    )


def execute_and_observe(user_input: str, auto_score: bool = True, persist: bool = True) -> AgentTrace:
    """
    Executes the agent and immediately passes the trace through the full
    observability pipeline: Anomaly Detection, Faithfulness Scoring, and
    Auditor Classification (if flagged).
    """
    trace = run_agent(user_input)
    trace_dict = trace.model_dump()

    # 1. Anomaly Scoring
    if auto_score:
        try:
            from ml.train_anomaly_model import predict_single_trace
            trace.anomaly_score = predict_single_trace(trace_dict)
        except Exception:
            pass

        # 2. Faithfulness Scoring
        try:
            from ml.faithfulness_scorer import score_trace
            f_score, _ = score_trace(trace_dict)
            trace.faithfulness_score = f_score
        except Exception:
            pass

        # 3. Auditor if flagged
        try:
            from ml.threshold_config import load_threshold
            from auditor.auditor import classify_trace
            threshold = load_threshold()
            if trace.anomaly_score is not None and trace.anomaly_score >= threshold:
                audit_res = classify_trace(trace.model_dump())
                trace.auditor_classification = AuditorClassification(
                    failure_mode=audit_res.get("failure_mode"),
                    explanation=audit_res.get("explanation"),
                )
        except Exception:
            pass

    # 4. Storage to DB
    if persist:
        try:
            from backend.database import SessionLocal, TraceRecord
            db = SessionLocal()
            try:
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
            finally:
                db.close()
        except Exception as exc:
            print(f"[storage warning] Could not persist trace to DB: {exc}")

    return trace


API_BASE_URL = os.environ.get("AGENTWATCH_API_URL", "http://127.0.0.1:8000")


def send_trace_to_api(trace: AgentTrace) -> bool:
    """POSTs a trace to the ingestion API."""
    import requests
    try:
        response = requests.post(
            f"{API_BASE_URL}/traces",
            data=trace.model_dump_json(),
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        return response.status_code in (200, 201)
    except Exception:
        return False


if __name__ == "__main__":
    t = execute_and_observe("I want a refund for order #4521, it never arrived.")
    print(t.model_dump_json(indent=2))
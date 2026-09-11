from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

class TraceStatus(str, Enum):
    success = "success"; error = "error"; incomplete = "incomplete"

class StepStatus(str, Enum):
    success = "success"; failure = "failure"; skipped = "skipped"

class FailureMode(str, Enum):
    wrong_tool_call = "wrong_tool_call"; hallucinated_success = "hallucinated_success"
    infinite_loop = "infinite_loop"; unauthorized_action = "unauthorized_action"
    low_confidence_answered_anyway = "low_confidence_answered_anyway"
    policy_violation = "policy_violation"; infrastructure_error = "infrastructure_error"; none = "none"

class Step(BaseModel):
    step_number: int
    decision_rationale: str
    tool_called: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[dict[str, Any]] = None
    step_status: StepStatus
    latency_ms: Optional[float] = None

class AuditorClassification(BaseModel):
    failure_mode: Optional[FailureMode] = None
    explanation: Optional[str] = None

class HumanReview(BaseModel):
    confirmed: Optional[bool] = None
    reviewer_note: Optional[str] = None

class AgentTrace(BaseModel):
    trace_id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)
    agent_name: str
    agent_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_input: str
    steps: list[Step]
    final_response: str
    status: TraceStatus
    total_latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    anomaly_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    auditor_classification: Optional[AuditorClassification] = None
    human_review: Optional[HumanReview] = None
    model_config = ConfigDict(use_enum_values=True)
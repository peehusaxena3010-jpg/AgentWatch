"""
AgentWatch — FastAPI Observability, Ingestion & Governance API.

Endpoints:
    GET  /health                        -> System health & component readiness
    GET  /stats                         -> Aggregated observability metrics & KPI stats
    POST /traces                        -> Ingest a new trace (with optional auto-scoring)
    GET  /traces                        -> List traces (with filtering & pagination)
    GET  /traces/{trace_id}             -> Retrieve complete execution trace by ID
    PATCH /traces/{trace_id}/review     -> Submit human review (confirm / reject)
    DELETE /traces/{trace_id}           -> Remove a trace

Pipeline Automation:
    POST /pipeline/run-agent            -> Execute live support agent & observe
    POST /pipeline/train-anomaly        -> Fit Isolation Forest model & score traces
    POST /pipeline/score-faithfulness   -> Score response faithfulness across traces
    POST /pipeline/run-auditor          -> Classify flagged traces with LLM auditor
    POST /pipeline/recalibrate          -> Recalibrate threshold via Youden's J statistic
    POST /pipeline/generate-dataset     -> Batch generate synthetic scenario traces

Run with:  uvicorn backend.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import init_db, get_db, TraceRecord, trace_record_to_dict
from backend.models.trace_models import AgentTrace, HumanReview
from ml.threshold_config import load_threshold

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="AgentWatch Observability & Governance API",
    description="ML-LLM Hybrid Observability and Governance Framework for Autonomous AI Agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# System Health & Summary Analytics
# --------------------------------------------------------------------------

@app.get("/health")
def get_health(db: Session = Depends(get_db)):
    """Reports system health, database record counts, and model status."""
    total_traces = db.query(TraceRecord).count()
    threshold = load_threshold()
    model_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "ml", "models", "anomaly_model.joblib")
    )
    return {
        "status": "healthy",
        "database_connected": True,
        "total_traces": total_traces,
        "active_flag_threshold": threshold,
        "anomaly_model_trained": model_exists,
    }


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Returns high-level KPI metrics, failure distributions, and review precision."""
    records = db.query(TraceRecord).all()
    total = len(records)
    threshold = load_threshold()

    if total == 0:
        return {
            "total_traces": 0,
            "success_count": 0,
            "error_count": 0,
            "incomplete_count": 0,
            "flagged_count": 0,
            "avg_latency_ms": 0.0,
            "avg_anomaly_score": 0.0,
            "avg_faithfulness_score": 0.0,
            "reviewed_count": 0,
            "confirmed_issues": 0,
            "false_positives": 0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0,
            "active_threshold": threshold,
            "failure_modes": {},
        }

    success_count = sum(1 for r in records if r.status == "success")
    error_count = sum(1 for r in records if r.status == "error")
    incomplete_count = sum(1 for r in records if r.status == "incomplete")

    flagged_count = sum(1 for r in records if (r.anomaly_score is not None and r.anomaly_score >= threshold))

    anomaly_scores = [r.anomaly_score for r in records if r.anomaly_score is not None]
    faith_scores = [r.faithfulness_score for r in records if r.faithfulness_score is not None]
    latencies = [r.total_latency_ms for r in records if r.total_latency_ms is not None]

    avg_anomaly = round(sum(anomaly_scores) / len(anomaly_scores), 3) if anomaly_scores else 0.0
    avg_faith = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    # Parse JSON for failure modes and human reviews
    failure_modes: dict[str, int] = {}
    reviewed_count = 0
    confirmed_issues = 0
    false_positives = 0

    flagged_reviewed = 0
    flagged_confirmed = 0

    for r in records:
        try:
            full = json.loads(r.full_trace_json)
        except Exception:
            continue

        audit = full.get("auditor_classification") or {}
        fm = audit.get("failure_mode")
        if fm and fm != "none":
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

        review = full.get("human_review") or {}
        if review.get("confirmed") is not None:
            reviewed_count += 1
            if review["confirmed"]:
                confirmed_issues += 1
            else:
                false_positives += 1

            if r.anomaly_score is not None and r.anomaly_score >= threshold:
                flagged_reviewed += 1
                if review["confirmed"]:
                    flagged_confirmed += 1

    fp_rate = round(false_positives / reviewed_count, 3) if reviewed_count else 0.0
    tpr = round(flagged_confirmed / confirmed_issues, 3) if confirmed_issues else 1.0

    return {
        "total_traces": total,
        "success_count": success_count,
        "error_count": error_count,
        "incomplete_count": incomplete_count,
        "flagged_count": flagged_count,
        "avg_latency_ms": avg_latency,
        "avg_anomaly_score": avg_anomaly,
        "avg_faithfulness_score": avg_faith,
        "reviewed_count": reviewed_count,
        "confirmed_issues": confirmed_issues,
        "false_positives": false_positives,
        "false_positive_rate": fp_rate,
        "true_positive_rate": tpr,
        "active_threshold": threshold,
        "failure_modes": failure_modes,
    }


# --------------------------------------------------------------------------
# Trace CRUD Endpoints
# --------------------------------------------------------------------------

@app.post("/traces", status_code=201)
def create_trace(trace: AgentTrace, db: Session = Depends(get_db)):
    """Stores a new agent trace, auto-calculating scores if not provided."""
    existing = db.get(TraceRecord, str(trace.trace_id))
    if existing:
        raise HTTPException(status_code=409, detail=f"Trace {trace.trace_id} already exists.")

    trace_dict = trace.model_dump()

    # Auto-score anomaly if absent
    if trace.anomaly_score is None:
        try:
            from ml.train_anomaly_model import predict_single_trace
            trace.anomaly_score = predict_single_trace(trace_dict)
        except Exception:
            pass

    # Auto-score faithfulness if absent
    if trace.faithfulness_score is None:
        try:
            from ml.faithfulness_scorer import score_trace
            f_score, _ = score_trace(trace_dict)
            trace.faithfulness_score = f_score
        except Exception:
            pass

    # Auto-audit if flagged or status != success
    threshold = load_threshold()
    should_audit = (
        (trace.anomaly_score is not None and trace.anomaly_score >= threshold)
        or (trace.status != "success")
    )
    if should_audit and not trace.auditor_classification:
        try:
            from auditor.auditor import classify_trace
            audit_res = classify_trace(trace.model_dump())
            trace.auditor_classification = {
                "failure_mode": audit_res.get("failure_mode"),
                "explanation": audit_res.get("explanation"),
            }
        except Exception:
            pass

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
    return {"trace_id": str(trace.trace_id), "stored": True, "anomaly_score": trace.anomaly_score, "faithfulness_score": trace.faithfulness_score}


@app.get("/traces")
def list_traces(
    status: Optional[str] = None,
    agent_name: Optional[str] = None,
    min_anomaly: Optional[float] = None,
    failure_mode: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Lists traces with filtering, search, and pagination."""
    query = db.query(TraceRecord)
    if status:
        query = query.filter(TraceRecord.status == status)
    if agent_name:
        query = query.filter(TraceRecord.agent_name == agent_name)
    if min_anomaly is not None:
        query = query.filter(TraceRecord.anomaly_score >= min_anomaly)

    records = query.order_by(TraceRecord.timestamp.desc()).offset(offset).limit(limit).all()

    results = []
    for r in records:
        full = json.loads(r.full_trace_json)
        audit = full.get("auditor_classification") or {}
        review = full.get("human_review") or {}

        # Filter by failure mode if specified
        if failure_mode and audit.get("failure_mode") != failure_mode:
            continue

        results.append({
            "trace_id": r.trace_id,
            "session_id": r.session_id,
            "agent_name": r.agent_name,
            "timestamp": r.timestamp,
            "status": r.status,
            "user_input": full.get("user_input", "")[:90],
            "num_steps": len(full.get("steps", [])),
            "anomaly_score": r.anomaly_score,
            "faithfulness_score": r.faithfulness_score,
            "total_latency_ms": r.total_latency_ms,
            "failure_mode": audit.get("failure_mode"),
            "reviewed": review.get("confirmed") is not None,
            "review_confirmed": review.get("confirmed"),
        })

    return results


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    """Fetches full trace detail (steps, tool calls, audit, review) by ID."""
    record = db.get(TraceRecord, trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace_record_to_dict(record)


@app.patch("/traces/{trace_id}/review")
def submit_human_review(trace_id: str, review: HumanReview, db: Session = Depends(get_db)):
    """Records human review decision (confirmed / rejected) to drive feedback calibration."""
    record = db.get(TraceRecord, trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found.")

    full_trace = json.loads(record.full_trace_json)
    full_trace["human_review"] = review.model_dump()
    record.full_trace_json = json.dumps(full_trace)
    db.commit()
    return {"trace_id": trace_id, "review_recorded": True, "confirmed": review.confirmed}


@app.delete("/traces/{trace_id}")
def delete_trace(trace_id: str, db: Session = Depends(get_db)):
    """Deletes a trace by ID."""
    record = db.get(TraceRecord, trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found.")
    db.delete(record)
    db.commit()
    return {"trace_id": trace_id, "deleted": True}


# --------------------------------------------------------------------------
# Pipeline Automation Endpoints
# --------------------------------------------------------------------------

class RunAgentRequest(BaseModel):
    user_input: str = Field(..., description="Query or prompt to execute through the agent")
    auto_score: bool = Field(True, description="Whether to compute anomaly, faithfulness, and audit scores")


@app.post("/pipeline/run-agent")
def pipeline_run_agent(req: RunAgentRequest):
    """Executes the AI support agent and observes the resulting execution trace."""
    from agents.support_agent import execute_and_observe
    trace = execute_and_observe(req.user_input, auto_score=req.auto_score, persist=True)
    return trace.model_dump()


@app.post("/pipeline/train-anomaly")
def pipeline_train_anomaly():
    """Trains Isolation Forest model on stored traces and updates anomaly scores."""
    from ml.train_anomaly_model import train_and_score
    result = train_and_score()
    return result


@app.post("/pipeline/score-faithfulness")
def pipeline_score_faithfulness():
    """Scores response faithfulness across all traces in the database."""
    from ml.faithfulness_scorer import FaithfulnessScorer
    updated = FaithfulnessScorer.score_and_update_db()
    return {"status": "success", "scored_count": updated}


@app.post("/pipeline/run-auditor")
def pipeline_run_auditor():
    """Runs the LLM/heuristic auditor over all flagged traces."""
    from auditor.run_auditor import run_audit
    run_audit()
    return {"status": "success", "message": "Auditor batch execution completed."}


@app.post("/pipeline/recalibrate")
def pipeline_recalibrate():
    """Executes the feedback loop to recalibrate the anomaly threshold using Youden's J."""
    from ml.feedback_loop import load_reviewed_traces, false_positive_rate_at_threshold, find_best_threshold, MIN_REVIEWS_REQUIRED
    from ml.threshold_config import load_threshold, save_threshold

    pairs = load_reviewed_traces()
    if len(pairs) < MIN_REVIEWS_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_REVIEWS_REQUIRED} human reviews are required to recalibrate. Current count: {len(pairs)}.",
        )

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
        "status": "recalibrated",
        "reviewed_traces_used": len(pairs),
        "old_threshold": round(old_threshold, 3),
        "old_fpr": round(old_fpr, 3),
        "old_tpr": round(old_tpr, 3),
        "new_threshold": round(new_threshold, 3),
        "new_fpr": round(new_fpr, 3),
        "new_tpr": round(new_tpr, 3),
        "youdens_j": round(best_j, 3),
        "improvement": round(old_fpr - new_fpr, 3),
    }


class GenerateDatasetRequest(BaseModel):
    sample_size: int = Field(10, ge=1, le=100)


@app.post("/pipeline/generate-dataset")
def pipeline_generate_dataset(req: GenerateDatasetRequest):
    """Generates randomized synthetic customer service scenario traces."""
    from scripts.generate_dataset import SCENARIO_POOL
    from agents.support_agent import execute_and_observe

    generated = 0
    for _ in range(req.sample_size):
        _, templates = random.choice(SCENARIO_POOL)
        prompt = random.choice(templates)
        execute_and_observe(prompt, auto_score=True, persist=True)
        generated += 1

    return {"status": "success", "generated_count": generated}


# --------------------------------------------------------------------------
# Built-in Modern Web Portal
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_web_portal():
    """Serves the modern AgentWatch web portal UI."""
    return HTMLResponse(content=_PORTAL_HTML)


_PORTAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgentWatch — AI Agent Observability</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            brand: { 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' }
          }
        }
      }
    }
  </script>
  <style>
    body { background-color: #0b0f19; color: #f3f4f6; }
    .card { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .status-badge { padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
  </style>
</head>
<body class="min-h-screen">
  <!-- Top Navigation -->
  <header class="border-b border-gray-800 bg-gray-950/80 sticky top-0 z-50 backdrop-blur">
    <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
          AW
        </div>
        <div>
          <span class="font-bold text-lg tracking-tight text-white">AgentWatch</span>
          <span class="text-xs text-indigo-400 ml-2 font-mono px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-800">Hybrid Observability</span>
        </div>
      </div>
      <div class="flex items-center space-x-4 text-sm">
        <span id="api-status" class="flex items-center text-emerald-400 text-xs font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse"></span>
          API Online
        </span>
        <a href="http://127.0.0.1:8501" target="_blank" class="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow transition flex items-center">
          Launch Streamlit UI ↗
        </a>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 font-medium text-xs transition">
          API Docs
        </a>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 space-y-8">
    <!-- Hero / Intro -->
    <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <h1 class="text-2xl font-bold text-white tracking-tight">AI Agent Observability & Governance</h1>
        <p class="text-gray-400 text-sm mt-1">Multi-lens monitoring: Unsupervised Isolation Forest, NLI Faithfulness, Behavioral LLM Auditor & Calibration.</p>
      </div>
      <div class="flex gap-2">
        <button onclick="runSampleTrace()" class="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow transition">
          + Run Test Agent
        </button>
        <button onclick="refreshData()" class="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs font-medium transition">
          ↻ Refresh
        </button>
      </div>
    </div>

    <!-- KPI Metric Cards -->
    <div class="grid grid-cols-2 md:grid-cols-6 gap-4">
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Total Traces</div>
        <div id="stat-total" class="text-2xl font-bold text-white mt-1">--</div>
      </div>
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Success Rate</div>
        <div id="stat-success" class="text-2xl font-bold text-emerald-400 mt-1">--</div>
      </div>
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Flagged Anomalies</div>
        <div id="stat-flagged" class="text-2xl font-bold text-amber-400 mt-1">--</div>
      </div>
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Avg Faithfulness</div>
        <div id="stat-faith" class="text-2xl font-bold text-cyan-400 mt-1">--</div>
      </div>
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Active Threshold</div>
        <div id="stat-thresh" class="text-2xl font-bold text-indigo-400 mt-1">--</div>
      </div>
      <div class="card p-4 rounded-xl">
        <div class="text-xs text-gray-400 font-medium">Reviews (FP Rate)</div>
        <div id="stat-reviews" class="text-2xl font-bold text-rose-400 mt-1">--</div>
      </div>
    </div>

    <!-- Trace Feed & Table -->
    <div class="card rounded-2xl overflow-hidden p-6 space-y-4">
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold text-white">Execution Traces Feed</h2>
        <span class="text-xs text-gray-400" id="trace-count-indicator">Loading traces...</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm text-gray-300">
          <thead class="text-xs uppercase bg-gray-900/60 text-gray-400 border-b border-gray-800 font-mono">
            <tr>
              <th class="py-3 px-4">Status</th>
              <th class="py-3 px-4">User Prompt</th>
              <th class="py-3 px-4">Steps</th>
              <th class="py-3 px-4">Latency</th>
              <th class="py-3 px-4">Anomaly Score</th>
              <th class="py-3 px-4">Faithfulness</th>
              <th class="py-3 px-4">Audited Failure</th>
              <th class="py-3 px-4">Action</th>
            </tr>
          </thead>
          <tbody id="trace-table-body" class="divide-y divide-gray-800 font-sans">
            <tr>
              <td colspan="8" class="text-center py-8 text-gray-500">Loading traces...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <script>
    async function refreshData() {
      try {
        const statsRes = await fetch('/stats');
        const stats = await statsRes.json();
        document.getElementById('stat-total').innerText = stats.total_traces;
        const sRate = stats.total_traces > 0 ? Math.round((stats.success_count / stats.total_traces) * 100) + '%' : '0%';
        document.getElementById('stat-success').innerText = sRate;
        document.getElementById('stat-flagged').innerText = stats.flagged_count;
        document.getElementById('stat-faith').innerText = (stats.avg_faithfulness_score * 100).toFixed(0) + '%';
        document.getElementById('stat-thresh').innerText = stats.active_threshold.toFixed(2);
        document.getElementById('stat-reviews').innerText = stats.reviewed_count + ' (' + Math.round(stats.false_positive_rate * 100) + '%)';

        const tracesRes = await fetch('/traces?limit=25');
        const traces = await tracesRes.json();
        document.getElementById('trace-count-indicator').innerText = 'Showing ' + traces.length + ' latest traces';

        const tbody = document.getElementById('trace-table-body');
        if (traces.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" class="text-center py-8 text-gray-500">No traces found. Click "+ Run Test Agent" to generate one.</td></tr>';
          return;
        }

        tbody.innerHTML = traces.map(t => {
          const statusBg = t.status === 'success' ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800' : 'bg-rose-950/80 text-rose-400 border border-rose-800';
          const aScore = t.anomaly_score !== null ? t.anomaly_score.toFixed(2) : '--';
          const aColor = t.anomaly_score >= stats.active_threshold ? 'text-amber-400 font-semibold' : 'text-gray-400';
          const fScore = t.faithfulness_score !== null ? (t.faithfulness_score * 100).toFixed(0) + '%' : '--';
          const fColor = t.faithfulness_score !== null && t.faithfulness_score < 0.7 ? 'text-rose-400 font-semibold' : 'text-gray-400';
          const fm = t.failure_mode ? '<span class="text-rose-400 bg-rose-950/50 px-2 py-0.5 rounded border border-rose-800 text-xs font-mono">' + t.failure_mode + '</span>' : '<span class="text-gray-500 text-xs">none</span>';

          return `
            <tr class="hover:bg-gray-800/40 transition">
              <td class="py-3 px-4"><span class="status-badge ${statusBg}">${t.status}</span></td>
              <td class="py-3 px-4 max-w-xs truncate text-white">${t.user_input}</td>
              <td class="py-3 px-4 font-mono text-xs">${t.num_steps}</td>
              <td class="py-3 px-4 font-mono text-xs text-gray-400">${t.total_latency_ms ? Math.round(t.total_latency_ms) + 'ms' : '--'}</td>
              <td class="py-3 px-4 font-mono text-xs ${aColor}">${aScore}</td>
              <td class="py-3 px-4 font-mono text-xs ${fColor}">${fScore}</td>
              <td class="py-3 px-4">${fm}</td>
              <td class="py-3 px-4">
                <a href="/traces/${t.trace_id}" target="_blank" class="text-xs text-indigo-400 hover:text-indigo-300 underline font-mono">
                  JSON ↗
                </a>
              </td>
            </tr>
          `;
        }).join('');
      } catch (err) {
        console.error('Failed to fetch data:', err);
      }
    }

    async function runSampleTrace() {
      const prompt = "I want a refund for order #4521, it never arrived.";
      try {
        await fetch('/pipeline/run-agent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_input: prompt, auto_score: true })
        });
        refreshData();
      } catch (err) {
        alert('Error: ' + err);
      }
    }

    window.onload = refreshData;
  </script>
</body>
</html>
"""

# AgentWatch

**An ML-LLM Hybrid Observability and Governance Framework for Autonomous AI Agents**

[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-SaaS_Dashboard-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation_Forest-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Auditor_%26_Agent-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![SQLite](https://img.shields.io/badge/SQLite-Persistent_Storage-003B57.svg?style=flat&logo=sqlite)](https://sqlite.org)

---

## 1. Project Overview

Autonomous AI agents reason across multiple steps, invoke external tools, and take real-world actions on a user's behalf. However, agents fail in subtle ways that conventional software monitoring (uptime, latency, HTTP 500 error codes) cannot catch:
- Silently executing the wrong tool or passing invalid arguments.
- Hallucinating a successful resolution (e.g. claiming a refund was issued when the payment gateway returned an error).
- Looping indefinitely between tools without making progress.
- Taking unauthorized actions or violating organizational policies.

**AgentWatch** provides the missing observability and governance layer purpose-built for autonomous agents. It combines four complementary evaluation lenses:
1. **Unsupervised Machine Learning (Isolation Forest)**: Detects statistically anomalous execution traces using 12 engineered behavioural features without requiring initial labelled failure data.
2. **Independent Response Faithfulness Scorer (NLI / Hallucination Detection)**: Treats collected tool results as the *premise* and the agent's final answer as the *hypothesis*, verifying whether claims are strictly grounded in collected evidence.
3. **Behavioral LLM Auditor**: Classifies flagged traces against a fixed 8-category failure taxonomy with natural-language root cause explanations.
4. **Closed Human-in-the-Loop Feedback Recalibration**: Uses reviewer confirmations and rejections to dynamically recalibrate the anomaly detection threshold using **Youden's J statistic** ($J = \text{TPR} - \text{FPR}$), reducing false positives over time while maintaining high recall.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│  Streamlit SaaS Dashboard (Port 8501)  &  FastAPI Web Portal (Port 8000)│
│  - KPI Metric Cards & Health Status    - Step-by-Step Visual Timeline   │
│  - Interactive Trace Inspector         - Live Agent Testing Workbench   │
│  - Human Review Confirmation / Reject  - Recalibration Curve (Youden's) │
└──────────────────┬─────────────────────────────────────┬───────────────┘
                   │ HTTP / Direct                       │ HTTP REST
                   ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI REST SERVICE                            │
│  Endpoints: /traces, /traces/{id}, /traces/{id}/review, /stats        │
│  Pipeline:  /pipeline/run-agent, /pipeline/train-anomaly,              │
│             /pipeline/score-faithfulness, /pipeline/run-auditor,       │
│             /pipeline/recalibrate, /pipeline/generate-dataset         │
└──────────────────┬─────────────────────────────────────┬───────────────┘
                   │ SQLAlchemy ORM                      │
                   ▼                                     ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       SQLITE PERSISTENT STORAGE      │  │        AI AGENT LOOP         │
│  - Indexed metadata columns          │  │  - Gemini 2.5 Flash / Mock   │
│  - Structured JSON trace payload     │  │  - Tools: check_order_status │
│  - Zero data loss across restarts    │  │    issue_refund, escalation  │
└──────────────────┬───────────────────┘  └──────────────┬───────────────┘
                   │                                     │
                   ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY & GOVERNANCE CORE                    │
│  1. Unsupervised Anomaly Detection (Isolation Forest, 12 Features)     │
│  2. Independent Faithfulness Scorer (NLI Evidence-Hypothesis Engine)   │
│  3. Behavioral LLM Auditor (8-Category Failure Taxonomy)               │
│  4. Closed-Loop Feedback Recalibration (Youden's J Threshold Search)   │
└────────────────────────────────────────────────────────────────────────┘
```

### Complete End-to-End Data Flow

```
User submits request
   │
   ▼
Support Agent executes multi-turn tool-calling loop (Gemini 2.5 Flash / Scripted)
   │  Logs decision rationale, tool calls, tool results, latency, and status
   ▼
Execution Trace generated (Schema-validated Pydantic model)
   │
   ▼
FastAPI Ingestion Service persists trace to SQLite (traces.db)
   │
   ├──▶ Lens 1: Isolation Forest extracts 12 numerical features ──▶ writes anomaly_score
   │
   ├──▶ Lens 2: Faithfulness Scorer evaluates evidence vs. claims ─▶ writes faithfulness_score
   │
   ├──▶ Lens 3: If anomaly_score ≥ threshold:
   │            LLM Auditor classifies failure mode + explanation ─▶ writes auditor_classification
   │
   ▼
Dashboard surfaces live KPI metrics, step timeline, and diagnostics
   │
   ▼
Human Reviewer inspects flagged trace and submits Confirm / Reject
   │
   ▼
Closed Feedback Loop evaluates candidate operating points using Youden's J ($J = TPR - FPR$)
   │
   ▼
Updated threshold persisted to threshold_config.json, automatically adopted by all components
```

---

## 3. Failure-Mode Taxonomy

Every flagged trace is audited and classified into one of eight standardized failure modes:

| Category | Description |
|---|---|
| `wrong_tool_call` | Inappropriate tool called, or right tool invoked with non-existent/invalid parameters given the context. |
| `hallucinated_success` | Final response claims success or asserts a refund was processed when tool executions returned errors or failed. |
| `infinite_loop` | Agent called the same tool with identical arguments repeatedly without making progress. |
| `unauthorized_action` | Action taken without required prerequisite checks (e.g. issuing a refund without checking order status). |
| `low_confidence_answered_anyway` | Answered definitively despite incomplete or ambiguous information. |
| `policy_violation` | Explicit violation of business logic rules (e.g. attempting a double refund on an already refunded order). |
| `infrastructure_error` | Failures caused by network timeouts, rate limits, or API outages rather than faulty agent reasoning. |
| `none` | Clean execution; reasoning, tool calls, and final response are fully aligned and supported. |

---

## 4. Feature Engineering for Anomaly Detection

The Isolation Forest model extracts **twelve behavioural and structural features** from each execution trace:

| # | Feature Name | Description & Observability Rationale |
|---|---|---|
| 1 | `num_steps` | Total reasoning and execution steps taken. |
| 2 | `total_latency_ms` | Cumulative execution time (log-transformed to mitigate extreme outlier skew). |
| 3 | `avg_step_latency_ms` | Average time spent per step (log-transformed). |
| 4 | `max_step_latency_ms` | Maximum step latency, identifying sudden stalls or retries (log-transformed). |
| 5 | `num_tool_calls` | Total number of tools invoked during the session. |
| 6 | `num_distinct_tools` | Diversity of tools utilized. |
| 7 | `repeated_consecutive_calls` | Count of adjacent duplicate tool calls (strong signal for infinite loops). |
| 8 | `num_failed_steps` | Count of steps where step status was recorded as failure. |
| 9 | `num_error_results` | Number of tool returns containing an explicit error payload. |
| 10 | `is_error_status` | Binary indicator for whether overall trace outcome was `error`. |
| 11 | `is_incomplete_status` | Binary indicator for whether the agent terminated prematurely without a final answer. |
| 12 | `final_response_len` | Character length of the final user-facing response. |

---

## 5. Technology Stack

- **Backend / API**: FastAPI, Uvicorn, Starlette, Pydantic v2
- **Persistent Database**: SQLite, SQLAlchemy ORM
- **Machine Learning**: scikit-learn (Isolation Forest), Pandas, NumPy, Joblib
- **Faithfulness & NLI Engine**: Custom evidence-premise hypothesis evaluation engine with entity and claim verification
- **LLM Agent & Auditor**: Google Gemini 2.5 Flash via `google-genai` SDK + scenario-aware mock engine
- **Dashboard & Frontend**: Streamlit, Plotly, Altair, Tailwind CSS / HTML5 web portal
- **Testing**: Python standard `unittest`, `fastapi.testclient`, `httpx`

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/S-amarth-K/AgentWatch.git
cd AgentWatch

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn sqlalchemy streamlit scikit-learn pandas pydantic python-dotenv requests httpx plotly altair
```

---

## 7. Environment Variables Reference

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(Optional)* | Google AI Studio API key for Gemini 2.5 Flash. If left empty or missing, AgentWatch runs in high-fidelity mock mode. |
| `MOCK_MODE` | `true` | When `true`, runs fast, zero-cost, scenario-aware agent and auditor executions. Set to `false` when using a real Gemini API key. |
| `AGENTWATCH_API_URL` | `http://127.0.0.1:8000` | Ingestion API base URL for agent traces and pipeline commands. |
| `DATABASE_URL` | `sqlite:///./traces.db` | SQLAlchemy connection string for persistent SQLite trace storage. |

---

## 8. Database Architecture & Schema

The database uses SQLite via SQLAlchemy ORM with a hybrid relational + document model:
- Commonly queried and filtered fields are indexed columns for high-speed dashboards.
- The complete step-by-step execution history (reasoning, tool args, raw results, auditor diagnosis, human review) is preserved as JSON in `full_trace_json`.

### Table Schema: `traces`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `trace_id` | `VARCHAR` | Primary Key, Indexed | UUID identifying the trace run. |
| `session_id` | `VARCHAR` | Indexed | UUID grouping traces within a user conversation session. |
| `agent_name` | `VARCHAR` | Indexed | Name of the agent (e.g. `support_agent`). |
| `agent_version` | `VARCHAR` | Nullable | Agent semantic version (e.g. `v1`). |
| `timestamp` | `DATETIME` | Indexed | UTC timestamp when trace was generated. |
| `status` | `VARCHAR` | Indexed | Execution status: `success`, `error`, or `incomplete`. |
| `anomaly_score` | `FLOAT` | Indexed, Nullable | Isolation Forest anomaly score $[0.0, 1.0]$. |
| `faithfulness_score` | `FLOAT` | Indexed, Nullable | NLI evidence grounding score $[0.0, 1.0]$. |
| `total_latency_ms` | `FLOAT` | Nullable | Total execution duration in milliseconds. |
| `full_trace_json` | `TEXT` | Not Null | Complete serialized JSON trace record. |

---

## 9. Running the Complete Application

### Option A: Launch All Services

Open two terminals:

**Terminal 1 — FastAPI Backend & Web Portal (Port 8000):**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API Documentation: `http://127.0.0.1:8000/docs`
- Built-in Web Portal: `http://127.0.0.1:8000/`

**Terminal 2 — Streamlit SaaS Observability Dashboard (Port 8501):**
```bash
streamlit run dashboard/app.py --server.port 8501
```
- Dashboard URL: `http://localhost:8501`

---

### Option B: Running the Pipeline via CLI Scripts

You can also run every component individually via the command line:

```bash
# 1. Generate 25 synthetic customer support scenario traces
python scripts/generate_dataset.py 25

# 2. Train Isolation Forest model & score all stored traces
python ml/train_anomaly_model.py

# 3. Evaluate response faithfulness across all traces
python ml/run_faithfulness_scorer.py

# 4. Run LLM auditor on flagged anomalous traces
python -m auditor.run_auditor

# 5. Check trace count in database
python check_traces.py

# 6. Check audited traces summary
python check_all_audited.py

# 7. Recalibrate anomaly threshold using accumulated human reviews
python ml/feedback_loop.py
```

---

## 10. Automated Testing & Verification

Run the comprehensive end-to-end automated test suite:

```bash
python -m unittest tests/test_full_pipeline.py -v
```

### Verified Test Cases:
1. `test_01_agent_execution_and_schema`: Executes agent tool-calling loop and validates output against schema.
2. `test_02_feature_extraction`: Validates extraction of all 12 engineered features.
3. `test_03_database_persistence_across_sessions`: Writes a trace, disconnects, opens a new database session, and verifies complete data persistence.
4. `test_04_faithfulness_scorer`: Verifies that grounded responses score high ($\ge 0.8$) while hallucinated responses are penalized ($< 0.5$).
5. `test_05_auditor_taxonomy_classification`: Confirms taxonomy classification for loops, policy violations, and wrong tool calls.
6. `test_06_human_feedback_and_youdens_j_recalibration`: Verifies threshold search maximizing Youden's J statistic on reviewed data.
7. `test_07_fastapi_endpoints`: Verifies REST API endpoints (`/health`, `/stats`, `/pipeline/run-agent`) using FastAPI client.

---

## 11. Requirement Traceability Matrix

| Requirement | Description | Status | Implemented In | Tested In |
|---|---|---|---|---|
| **FR-01** | Submit Agent Request | ✅ Implemented | `agents/support_agent.py`, `backend/main.py` (`/pipeline/run-agent`) | `tests/test_full_pipeline.py` |
| **FR-02** | Generate Execution Trace | ✅ Implemented | `agents/support_agent.py` | `tests/test_full_pipeline.py` |
| **FR-03** | Validate Execution Trace | ✅ Implemented | `backend/models/trace_models.py`, `schema.json` | `tests/test_full_pipeline.py` |
| **FR-04** | Store Execution Trace | ✅ Implemented | `backend/database.py`, `backend/main.py` (`POST /traces`) | `tests/test_full_pipeline.py` |
| **FR-05** | Retrieve Traces | ✅ Implemented | `backend/main.py` (`GET /traces`, `GET /traces/{id}`) | `tests/test_full_pipeline.py` |
| **FR-06** | Extract Behavioural Features | ✅ Implemented | `ml/feature_extraction.py` (12 features) | `tests/test_full_pipeline.py` |
| **FR-07** | Detect Anomalous Behaviour | ✅ Implemented | `ml/train_anomaly_model.py` (Isolation Forest) | `tests/test_full_pipeline.py` |
| **FR-08** | Flag Suspicious Traces | ✅ Implemented | `ml/threshold_config.py`, `backend/main.py` | `tests/test_full_pipeline.py` |
| **FR-09** | Classify Agent Behaviour | ✅ Implemented | `auditor/auditor.py`, `auditor/taxonomy.py` | `tests/test_full_pipeline.py` |
| **FR-10** | Evaluate Response Faithfulness | ✅ Implemented | `ml/faithfulness_scorer.py` | `tests/test_full_pipeline.py` |
| **FR-11** | Generate Faithfulness Score | ✅ Implemented | `ml/faithfulness_scorer.py`, `backend/main.py` | `tests/test_full_pipeline.py` |
| **FR-12** | Display Monitoring Results | ✅ Implemented | `dashboard/app.py` (Overview, Charts, KPIs) | Browser verification on port 8501 |
| **FR-13** | Inspect Execution Trace | ✅ Implemented | `dashboard/app.py` (Visual Step Timeline & JSON) | Browser verification on port 8501 |
| **FR-14** | Filter Traces | ✅ Implemented | `dashboard/app.py`, `backend/main.py` | `tests/test_full_pipeline.py` |
| **FR-15** | Record Human Review | ✅ Implemented | `backend/main.py` (`PATCH /traces/{id}/review`), `dashboard/app.py` | `tests/test_full_pipeline.py` |
| **FR-16** | Process Human Feedback | ✅ Implemented | `ml/feedback_loop.py` | `tests/test_full_pipeline.py` |
| **FR-17** | Recalibrate Detection Threshold | ✅ Implemented | `ml/feedback_loop.py` (Youden's J statistic) | `tests/test_full_pipeline.py` |
| **FR-18** | Persist Updated Threshold | ✅ Implemented | `ml/threshold_config.py` (`threshold_config.json`) | `tests/test_full_pipeline.py` |
| **FR-19** | Support Failure Scenarios | ✅ Implemented | `scripts/generate_dataset.py`, `agents/support_agent.py` | `tests/test_full_pipeline.py` |
| **FR-20** | Handle Infrastructure Errors | ✅ Implemented | `auditor/taxonomy.py`, `auditor/auditor.py` | `tests/test_full_pipeline.py` |

---

## 12. Authors & Academic Context

- **Samarth Khandelwal** (Registration: 23BIT0051)
- **Nakul Thombare** (Registration: 23BIT0013)
- **Faculty Guide**: Dr. Jerart Julus
- **School**: School of Computer Science Engineering and Information Systems
- **Department**: Department of Information Technology
- **Course**: BITE497J — Project - I
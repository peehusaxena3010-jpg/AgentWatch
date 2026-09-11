"""
AgentWatch — AI Agent Observability & Governance Dashboard.
Production-grade modern SaaS monitoring interface.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dashboard.data import (
    load_traces_df,
    get_full_trace,
    submit_human_review,
    get_summary_stats,
    execute_agent_test,
    train_anomaly_model_action,
    score_faithfulness_action,
    run_auditor_action,
    recalibrate_threshold_action,
    generate_scenarios_action,
)
from ml.threshold_config import load_threshold

# Page configuration
st.set_page_config(
    page_title="AgentWatch | AI Agent Observability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Design System (Inter, Glassmorphism, Modern SaaS Theme)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Global typography & base styling */
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
  }

  /* Metric cards */
  .kpi-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
  }
  .kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
  }
  .kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
  }

  /* Step execution cards in inspector */
  .step-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }
  .step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .step-number {
    font-size: 0.8rem;
    font-weight: 600;
    color: #818cf8;
    background: rgba(99, 102, 241, 0.15);
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid rgba(99, 102, 241, 0.3);
  }
  .step-tool {
    font-size: 0.85rem;
    font-weight: 600;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
  }
  .step-latency {
    font-size: 0.75rem;
    color: #94a3b8;
  }

  /* Badges */
  .badge-success {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-error {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: System Status & Quick Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **AgentWatch**")
    st.caption("ML-LLM Hybrid Observability & Governance")
    st.divider()

    current_threshold = load_threshold()
    st.markdown(f"**Flagging Threshold:** `{current_threshold:.3f}`")
    st.caption("Calibrated using Youden's J statistic from reviewer feedback.")

    st.divider()
    st.markdown("##### Quick Actions")
    if st.button("↻ Refresh Data", use_container_width=True):
        st.rerun()

    if st.button("⚡ Score Faithfulness", use_container_width=True):
        with st.spinner("Scoring response faithfulness..."):
            count = score_faithfulness_action()
            st.success(f"Scored {count} traces!")
            st.rerun()

    if st.button("🤖 Run LLM Auditor", use_container_width=True):
        with st.spinner("Auditing flagged traces..."):
            run_auditor_action()
            st.success("Auditor execution completed!")
            st.rerun()

    st.divider()
    st.markdown("##### Architecture Lenses")
    st.markdown("""
    - **Lens 1:** Isolation Forest (12 Features)
    - **Lens 2:** NLI Faithfulness Scorer
    - **Lens 3:** Behavioral LLM Auditor
    - **Lens 4:** Closed Youden's J Feedback Loop
    """)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
df = load_traces_df()
stats = get_summary_stats(df)

# Top Bar Header
st.title("🛡️ AgentWatch Dashboard")
st.markdown(
    "Continuous observability and governance framework for autonomous AI agents — combining "
    "**unsupervised machine learning**, **natural language inference**, and **LLM behavioral auditing**."
)

if df.empty:
    st.warning("⚠️ No execution traces found in SQLite storage. Click below to generate initial synthetic scenarios.")
    if st.button("🚀 Generate 12 Initial Synthetic Traces", type="primary"):
        with st.spinner("Generating traces..."):
            generate_scenarios_action(12)
            train_anomaly_model_action()
            score_faithfulness_action()
            run_auditor_action()
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------------------------
tab_overview, tab_explorer, tab_studio, tab_calibration, tab_pipeline = st.tabs([
    "📊 Executive Overview",
    "🔍 Trace Explorer & Inspector",
    "⚡ Live Agent Studio",
    "🎯 Model Governance & Recalibration",
    "⚙️ Dataset & Pipeline Ops",
])

# ===========================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ===========================================================================
with tab_overview:
    st.markdown("### Observability KPIs")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Total Traces</div>
          <div class="kpi-value">{stats['total_traces']}</div>
          <div class="kpi-sub">Stored in SQLite</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Success Rate</div>
          <div class="kpi-value" style="color: #34d399;">{stats['success_rate']}%</div>
          <div class="kpi-sub">{stats['total_traces'] - stats['error_count']} clean executions</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Flagged Anomalies</div>
          <div class="kpi-value" style="color: #fbbf24;">{stats['flagged_count']}</div>
          <div class="kpi-sub">Score ≥ {current_threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Avg Faithfulness</div>
          <div class="kpi-value" style="color: #38bdf8;">{stats['avg_faithfulness_score'] * 100:.0f}%</div>
          <div class="kpi-sub">NLI evidence grounding</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        audited_count = int((df["failure_mode"].notna() & (df["failure_mode"] != "none")).sum())
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Audited Failures</div>
          <div class="kpi-value" style="color: #f43f5e;">{audited_count}</div>
          <div class="kpi-sub">Taxonomy classified</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col6:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Reviewed (FPR)</div>
          <div class="kpi-value" style="color: #a855f7;">{stats['reviewed_count']} <span style="font-size: 1rem;">({stats['false_positive_rate']*100:.0f}%)</span></div>
          <div class="kpi-sub">Human-in-the-loop</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart row 1
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Trace Outcomes by Status")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.bar_chart(status_counts.set_index("Status"), color="#6366f1")

    with c2:
        st.subheader("Audited Failure Taxonomy Breakdown")
        failure_counts = df[df["failure_mode"].notna() & (df["failure_mode"] != "none")]["failure_mode"].value_counts().reset_index()
        failure_counts.columns = ["Failure Mode", "Count"]
        if failure_counts.empty:
            st.info("No failure modes detected yet. Run the LLM auditor on flagged traces.")
        else:
            st.bar_chart(failure_counts.set_index("Failure Mode"), color="#f43f5e")

    # Chart row 2
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Anomaly Score Distribution")
        score_series = df["anomaly_score"].dropna()
        if not score_series.empty:
            st.line_chart(score_series.sort_values().reset_index(drop=True), color="#fbbf24")
            st.caption(f"Horizontal reference line: calibrated flagging threshold = {current_threshold:.3f}")

    with c4:
        st.subheader("NLI Faithfulness Score vs. Anomaly Score")
        scatter_df = df[["anomaly_score", "faithfulness_score", "status"]].dropna()
        if not scatter_df.empty:
            st.scatter_chart(scatter_df, x="anomaly_score", y="faithfulness_score", color="status")
            st.caption("Quadrant separation: High Anomaly + Low Faithfulness represents critical multi-lens failures.")

    st.divider()

    # Recent Alerts Feed
    st.subheader("🚨 Priority Governance Alert Feed")
    flagged_df = df[df["anomaly_score"] >= current_threshold].head(5)
    if flagged_df.empty:
        st.success("✅ No traces currently exceed the anomaly threshold.")
    else:
        for _, row in flagged_df.iterrows():
            fm = row.get("failure_mode") or "Pending Audit"
            st.markdown(f"""
            <div style="background: rgba(244, 63, 94, 0.08); border-left: 4px solid #f43f5e; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 600; color: #f8fafc; font-size: 0.9rem;">Trace <code>{row['trace_id'][:12]}...</code></span>
                <span class="badge-warning">Anomaly: {row['anomaly_score']:.2f}</span>
                <span class="badge-error">{fm}</span>
              </div>
              <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">Prompt: "{row['user_input']}"</div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2: TRACE EXPLORER & INSPECTOR
# ===========================================================================
with tab_explorer:
    st.subheader("Trace Repository")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.5, 2])
    with fc1:
        status_filter = st.multiselect("Filter by Status", options=sorted(df["status"].dropna().unique()), default=[])
    with fc2:
        f_modes = sorted(df["failure_mode"].dropna().unique())
        failure_filter = st.multiselect("Filter by Failure Mode", options=f_modes, default=[])
    with fc3:
        min_anomaly = st.slider("Min Anomaly Score", 0.0, 1.0, 0.0, 0.05)
    with fc4:
        search_query = st.text_input("Search prompt / order ID", "")

    filtered_df = df.copy()
    if status_filter:
        filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
    if failure_filter:
        filtered_df = filtered_df[filtered_df["failure_mode"].isin(failure_filter)]
    filtered_df = filtered_df[filtered_df["anomaly_score"].fillna(0) >= min_anomaly]
    if search_query:
        filtered_df = filtered_df[filtered_df["user_input"].str.contains(search_query, case=False, na=False)]

    st.caption(f"Showing **{len(filtered_df)}** of **{len(df)}** execution traces")

    # Table view
    display_cols = [
        "trace_id", "timestamp", "status", "user_input", "num_steps",
        "anomaly_score", "faithfulness_score", "failure_mode", "reviewed"
    ]
    st.dataframe(
        filtered_df[display_cols].style.format({
            "anomaly_score": "{:.3f}",
            "faithfulness_score": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Detailed Step-by-Step Trace Inspector
    st.subheader("🔬 Deep Trace Inspector & Human Review")

    trace_options = [""] + filtered_df["trace_id"].tolist()
    selected_trace_id = st.selectbox(
        "Select a trace to inspect complete step reasoning, evidence, and audit logs:",
        options=trace_options,
        index=1 if len(trace_options) > 1 else 0,
    )

    if selected_trace_id:
        trace_data = get_full_trace(selected_trace_id)
        if not trace_data:
            st.error("Trace details not found.")
        else:
            # Header info card
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom: 20px;">
              <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                  <h4 style="margin: 0; color: #f8fafc;">Trace ID: <code>{trace_data.get('trace_id')}</code></h4>
                  <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                    Session: <code>{trace_data.get('session_id')}</code> | Agent: <b>{trace_data.get('agent_name')}</b> ({trace_data.get('agent_version')})
                  </div>
                </div>
                <div>
                  <span class="badge-{'success' if trace_data.get('status') == 'success' else 'error'}">{trace_data.get('status')}</span>
                </div>
              </div>
              <div style="margin-top: 14px; padding: 10px 14px; background: rgba(0,0,0,0.3); border-radius: 8px;">
                <b style="color: #cbd5e1;">User Prompt:</b> <span style="color: #f8fafc;">"{trace_data.get('user_input')}"</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Step-by-step visual execution flow
            st.markdown("#### 🔄 Step-by-Step Reasoning Flow")
            steps = trace_data.get("steps", [])
            if not steps:
                st.info("No tool steps recorded for this trace.")
            else:
                for s in steps:
                    step_num = s.get("step_number")
                    tool = s.get("tool_called") or "Final Response Generation"
                    status = s.get("step_status", "success")
                    latency = s.get("latency_ms", 0)
                    rationale = s.get("decision_rationale", "")
                    tool_args = s.get("tool_args")
                    tool_result = s.get("tool_result")

                    st.markdown(f"""
                    <div class="step-box">
                      <div class="step-header">
                        <div>
                          <span class="step-number">Step {step_num}</span>
                          <span class="step-tool" style="margin-left: 8px;">{tool}</span>
                        </div>
                        <div>
                          <span class="badge-{'success' if status == 'success' else 'error'}">{status}</span>
                          <span class="step-latency" style="margin-left: 8px;">⏱️ {latency} ms</span>
                        </div>
                      </div>
                      <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;">
                        <b>Reasoning:</b> {rationale}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if tool_args or tool_result:
                        with st.expander(f"View Tool Arguments & Result for Step {step_num}"):
                            c_arg, c_res = st.columns(2)
                            with c_arg:
                                st.caption("Tool Arguments:")
                                st.json(tool_args or {})
                            with c_res:
                                st.caption("Tool Execution Result:")
                                st.json(tool_result or {})

            # Final response
            st.markdown("#### 💬 Final Agent Response")
            st.info(trace_data.get("final_response") or "(No final response produced)")

            # Multi-Lens Observability Diagnostics
            st.markdown("#### 🔍 Observability & Governance Diagnostics")
            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                st.markdown("##### Lens 1: Statistical Anomaly Detection")
                a_score = trace_data.get("anomaly_score")
                if a_score is not None:
                    is_flagged = a_score >= current_threshold
                    st.metric("Anomaly Score", f"{a_score:.3f}", delta=f"{'FLAGGED' if is_flagged else 'NORMAL'}", delta_color="inverse" if is_flagged else "normal")
                    st.progress(float(a_score))
                    st.caption(f"Calibrated Threshold: {current_threshold:.3f}")
                else:
                    st.caption("Anomaly score not computed for this trace.")

            with diag_col2:
                st.markdown("##### Lens 2: Response Faithfulness (NLI)")
                f_score = trace_data.get("faithfulness_score")
                if f_score is not None:
                    st.metric("Faithfulness Score", f"{f_score * 100:.0f}%", delta=f"{'GROUNDED' if f_score >= 0.7 else 'LOW FAITHFULNESS'}")
                    st.progress(float(f_score))
                    st.caption("Assesses whether claims are strictly grounded in tool evidence.")
                else:
                    st.caption("Faithfulness score pending evaluation.")

            # LLM Auditor Classification
            st.markdown("##### Lens 3: Behavioral LLM Auditor Classification")
            audit = trace_data.get("auditor_classification")
            if audit and audit.get("failure_mode"):
                fm = audit.get("failure_mode")
                st.markdown(f"""
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px 16px;">
                  <b style="color: #fb7185;">Failure Mode:</b> <code>{fm}</code><br>
                  <span style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px; display: block;"><b>Root Cause Explanation:</b> {audit.get('explanation')}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No failure mode identified by the auditor for this trace.")

            # Human Review & Closed Feedback Loop
            st.markdown("---")
            st.markdown("#### ✍️ Human-in-the-Loop Review (Closed Feedback Loop)")
            existing_review = trace_data.get("human_review") or {}

            if existing_review.get("confirmed") is not None:
                is_conf = existing_review["confirmed"]
                st.success(f"✅ Already Reviewed: **{'Confirmed Real Failure' if is_conf else 'Rejected as False Positive'}**"
                           + (f" — Note: {existing_review.get('reviewer_note')}" if existing_review.get('reviewer_note') else ""))
            else:
                st.caption("Provide ground-truth feedback to recalibrate the anomaly detection threshold using Youden's J statistic.")
                note_input = st.text_input("Reviewer notes / rationale (optional):", key=f"note_{selected_trace_id}")
                btn_c1, btn_c2 = st.columns(2)

                with btn_c1:
                    if st.button("✅ Confirm Real Issue", key=f"conf_{selected_trace_id}", use_container_width=True):
                        submit_human_review(selected_trace_id, True, note_input)
                        st.success("Recorded: Confirmed real issue!")
                        st.rerun()

                with btn_c2:
                    if st.button("❌ Reject (False Positive)", key=f"rej_{selected_trace_id}", use_container_width=True):
                        submit_human_review(selected_trace_id, False, note_input)
                        st.warning("Recorded: False positive!")
                        st.rerun()


# ===========================================================================
# TAB 3: LIVE AGENT STUDIO / PLAYGROUND
# ===========================================================================
with tab_studio:
    st.subheader("⚡ Live Support Agent Testing Workbench")
    st.markdown("Execute the autonomous customer support agent in real time, observe its decision making, and inspect the resulting execution trace.")

    # Preset scenarios
    scenario_options = {
        "Custom Input": "",
        "Valid Refund (Overdue Order #4521)": "I want a refund for order #4521, it never arrived.",
        "Double Refund / Policy Violation (Order #3003)": "I'd like a refund for order #3003, it arrived broken.",
        "Non-Existent Order (Order #9999)": "Where is my order #9999? Please refund it.",
        "Delivered Order Check (Order #1001)": "Can you check the delivery status of order #1001?",
        "Processing Order Check (Order #2002)": "When will my order #2002 be delivered?",
        "Loop Trigger Scenario": "Please loop-verify order #4521 status thoroughly and double check.",
        "Ambiguous Request (No Order ID)": "My order is messed up, please fix it immediately.",
    }

    selected_scenario = st.selectbox("Choose a scenario or test case:", options=list(scenario_options.keys()))
    default_text = scenario_options[selected_scenario]

    prompt_input = st.text_area("User Request Prompt:", value=default_text, height=100)

    if st.button("🚀 Execute Agent & Observe Trace", type="primary"):
        if not prompt_input.strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Autonomous agent executing multi-turn tool-calling loop..."):
                trace_res = execute_agent_test(prompt_input.strip())
                st.success(f"Agent execution complete! Status: **{trace_res.get('status')}** (Latency: {trace_res.get('total_latency_ms')} ms)")

                # Immediate diagnostics
                r_c1, r_c2, r_c3 = st.columns(3)
                with r_c1:
                    a_sc = trace_res.get("anomaly_score", 0.0)
                    st.metric("Anomaly Score", f"{a_sc:.3f}")
                with r_c2:
                    f_sc = trace_res.get("faithfulness_score", 1.0)
                    st.metric("Faithfulness Score", f"{f_sc * 100:.0f}%")
                with r_c3:
                    audit_info = trace_res.get("auditor_classification") or {}
                    raw_fm = audit_info.get("failure_mode")
                    fm_name = getattr(raw_fm, "value", raw_fm) or "none"
                    st.metric("Audited Failure", fm_name)

                if audit_info.get("explanation"):
                    st.warning(f"**Auditor Explanation:** {audit_info['explanation']}")

                st.markdown(f"**Agent Response:** {trace_res.get('final_response')}")

                with st.expander("View Full Executed Trace JSON"):
                    st.json(trace_res)


# ===========================================================================
# TAB 4: GOVERNANCE & MODEL RECALIBRATION
# ===========================================================================
with tab_calibration:
    st.subheader("🎯 Closed Feedback Loop: Threshold Recalibration")
    st.markdown("""
    AgentWatch improves its detection accuracy over time using confirmed and rejected human reviews.
    The calibration engine searches candidate anomaly-score thresholds and selects the one that maximizes
    **Youden's J statistic**:
    $$J = \\text{True Positive Rate (Sensitivity)} - \\text{False Positive Rate (1 - Specificity)}$$
    """)

    from ml.feedback_loop import load_reviewed_traces, false_positive_rate_at_threshold

    pairs = load_reviewed_traces()
    st.markdown(f"**Reviewed Ground-Truth Traces Available:** `{len(pairs)}`")

    if len(pairs) < 5:
        st.warning("⚠️ At least 5 human reviews are required to run statistically meaningful recalibration. Review more traces in the **Trace Explorer** tab.")
    else:
        # Candidate threshold evaluation
        candidate_thresholds = sorted(set([p[0] for p in pairs]))
        rows = []
        for t in candidate_thresholds:
            fpr, tpr = false_positive_rate_at_threshold(pairs, t)
            j = tpr - fpr
            rows.append({
                "Threshold": round(t, 3),
                "False Positive Rate (FPR)": round(fpr, 3),
                "True Positive Rate (TPR)": round(tpr, 3),
                "Youden's J (TPR - FPR)": round(j, 3),
            })

        cand_df = pd.DataFrame(rows)
        st.markdown("##### Candidate Threshold Trade-off Table")
        st.dataframe(cand_df, use_container_width=True, hide_index=True)

        # Plot Youden's J curve
        st.markdown("##### Youden's J vs. Anomaly Threshold")
        chart_df = cand_df.set_index("Threshold")[["Youden's J (TPR - FPR)", "False Positive Rate (FPR)", "True Positive Rate (TPR)"]]
        st.line_chart(chart_df)

        st.markdown("---")
        if st.button("🎯 Recalibrate Anomaly Flagging Threshold Now", type="primary"):
            with st.spinner("Searching optimal operating point..."):
                res = recalibrate_threshold_action()
                if res["success"]:
                    st.success(
                        f"✅ Recalibration successful! New threshold: **{res['new_threshold']}** "
                        f"(FPR dropped from {res['old_fpr']*100:.0f}% to {res['new_fpr']*100:.0f}%, "
                        f"Recall: {res['new_tpr']*100:.0f}%, Youden's J = {res['youdens_j']:.3f})"
                    )
                    st.rerun()
                else:
                    st.error(res["message"])


# ===========================================================================
# TAB 5: DATASET & PIPELINE CONTROLS
# ===========================================================================
with tab_pipeline:
    st.subheader("⚙️ Dataset & Pipeline Orchestration")
    st.markdown("Batch operations for synthetic data generation, model fitting, and database management.")

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st.markdown("##### 1. Generate Synthetic Scenario Traces")
        st.caption("Creates randomized customer support traces across valid refunds, loop triggers, and policy violations.")
        sample_size = st.slider("Batch Size", 5, 50, 15)
        if st.button(f"Generate {sample_size} Scenarios", use_container_width=True):
            with st.spinner("Generating scenario traces..."):
                created = generate_scenarios_action(sample_size)
                st.success(f"Generated and stored {created} traces!")
                st.rerun()

        st.markdown("##### 2. Retrain Isolation Forest Model")
        st.caption("Fits unsupervised anomaly detection model on all stored traces with 12 engineered features.")
        if st.button("Train Isolation Forest Model", use_container_width=True):
            with st.spinner("Fitting Isolation Forest..."):
                res = train_anomaly_model_action()
                st.success(f"Model trained on {res.get('total_scored', 0)} traces! Flagged: {res.get('flagged_count', 0)}")
                st.rerun()

    with p_col2:
        st.markdown("##### 3. Batch Faithfulness Scorer")
        st.caption("Evaluates response evidence grounding across all traces in the database.")
        if st.button("Score Faithfulness for All Traces", use_container_width=True):
            with st.spinner("Evaluating faithfulness..."):
                count = score_faithfulness_action()
                st.success(f"Updated faithfulness scores for {count} traces!")
                st.rerun()

        st.markdown("##### 4. Batch LLM Auditor")
        st.caption("Runs taxonomy failure classification over all flagged anomalous traces.")
        if st.button("Audit Flagged Traces", use_container_width=True):
            with st.spinner("Running auditor..."):
                run_auditor_action()
                st.success("Auditor execution completed!")
                st.rerun()

    st.divider()
    st.markdown("##### Database Storage Information")
    st.code(f"Database File: {os.path.abspath('traces.db')}\nTotal Records: {len(df)}", language="bash")
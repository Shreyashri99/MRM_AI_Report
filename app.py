"""
MRM Auditor — Streamlit Dashboard
Interactive UI for the full Model Risk Governance Engine
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="AI Model Risk Governance Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1a1a2e;}
    .metric-card {background:#f8f9fa; border-left:4px solid #0066cc; padding:1rem; border-radius:6px; margin:0.5rem 0;}
    .risk-high   {color:#dc3545; font-weight:700;}
    .risk-medium {color:#fd7e14; font-weight:700;}
    .risk-low    {color:#198754; font-weight:700;}
    .verdict-box {border:2px solid; padding:1rem; border-radius:8px; text-align:center; font-size:1.2rem; font-weight:700;}
    .stProgress > div > div {background-color:#0066cc;}
    div[data-testid="stExpander"] {border:1px solid #dee2e6; border-radius:6px;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=60)
    st.title("MRM Auditor")
    st.caption("SR 11-7 / SS1-23 Compliant")
    st.divider()

    st.subheader("⚙️ Audit Configuration")
    model_type = st.selectbox(
        "Model Type", ["gradient_boosting", "logistic_regression"],
        index=0, help="Model architecture to train and audit"
    )
    drift_intensity = st.slider(
        "Drift Intensity", 0.0, 1.0, 0.3, 0.05,
        help="Higher = more severe simulated production drift"
    )
    api_key = st.text_input(
        "Anthropic API Key (optional)",
        type="password",
        help="Add your key to generate AI-powered narrative report"
    )

    run_btn = st.button("🚀 Run Full Audit", type="primary", use_container_width=True)
    st.divider()
    st.caption("**Phases:**\n1. Data Loading\n2. Bias Analysis\n3. Drift Detection\n4. Explainability\n5. SR 11-7 Scoring\n6. Report Generation")


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 AI Model Risk Governance Engine</p>', unsafe_allow_html=True)
st.markdown("**SR 11-7 / SS1-23 Automated Model Risk Auditor** — *Credit Scoring Model v1.0*")
st.divider()


# ── State ──────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    from pipeline import run_full_audit
    with st.spinner("⏳ Running full MRM audit (this takes ~60-90s for SHAP)..."):
        try:
            results = run_full_audit(
                model_type=model_type,
                drift_intensity=drift_intensity,
                use_cache=False,
                api_key=api_key,
            )
            st.session_state.results = results
            st.success("✅ Audit complete!")
        except Exception as e:
            st.error(f"❌ Audit failed: {e}")
            st.exception(e)


# ── If no results yet ──────────────────────────────────────────────────────
if st.session_state.results is None:
    st.info("👈 Configure settings in the sidebar and click **Run Full Audit** to begin.")

    st.markdown("### 📌 What this tool does")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**🔍 Phase 1–2**\nLoads the UCI Adult Income dataset, trains a credit scoring model, and runs full bias/fairness analysis across protected attributes (gender, race).")
    with cols[1]:
        st.markdown("**📡 Phase 3–4**\nSimulates production data drift, measures PSI and KS statistics per feature, and generates SHAP explainability analysis.")
    with cols[2]:
        st.markdown("**📋 Phase 5–6**\nScores the model against SR 11-7 / SS1-23 across 6 dimensions and generates a formal Model Risk Management Report.")
    st.stop()


# ── Load results ───────────────────────────────────────────────────────────
R = st.session_state.results
bm     = R["baseline_metrics"]
bias   = R["bias_results"]
drift  = R["drift_report"]
sr117  = R["sr117"]
mrm    = R["mrm_report"]
shap_i = R["shap_importance"]
fi     = R["feature_importance"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — VERDICT BANNER
# ══════════════════════════════════════════════════════════════════════════════
verdict = mrm.get("model_verdict", "N/A")
color_map = {
    "APPROVED":                  ("#198754", "#d1e7dd"),
    "APPROVED WITH CONDITIONS":  ("#fd7e14", "#fff3cd"),
    "RESTRICTED USE":            ("#dc3545", "#f8d7da"),
    "NOT APPROVED":              ("#dc3545", "#f8d7da"),
}
txt_color, bg_color = color_map.get(verdict, ("#6c757d", "#f8f9fa"))

st.markdown(f"""
<div style='background:{bg_color}; border:2px solid {txt_color}; padding:1.2rem; border-radius:10px; text-align:center;'>
    <span style='color:{txt_color}; font-size:1.4rem; font-weight:800;'>MODEL VERDICT: {verdict}</span><br>
    <span style='color:#444; font-size:0.95rem;'>{mrm.get("verdict_rationale","")}</span>
</div>
""", unsafe_allow_html=True)
st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SCORECARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Audit Scorecard")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AUC-ROC",        f"{bm['auc_roc']:.4f}",   help="Area Under ROC Curve on holdout test set")
c2.metric("Accuracy",       f"{bm['accuracy']:.1%}")
c3.metric("SR 11-7 Score",  f"{sr117.overall_score:.2f}/5", delta=f"{sr117.risk_tier}")
c4.metric("AUC Drop (Drift)", f"{drift.auc_drop:+.4f}", delta_color="inverse")
high_bias_n = sum(1 for r in bias if r.risk_level == "HIGH")
c5.metric("High Bias Flags", high_bias_n, delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋 SR 11-7 Scores",
    "🔍 Bias Analysis",
    "📡 Drift Detection",
    "🔬 Explainability",
    "📄 MRM Report",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — SR 11-7 SCORES
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown(f"### SR 11-7 Compliance: **{sr117.overall_score:.2f}/5.00** — {sr117.tier_label}")

    # Radar chart
    dim_names  = [d.name for d in sr117.dimensions]
    dim_scores = [d.score for d in sr117.dimensions]

    fig_radar = go.Figure(go.Scatterpolar(
        r=dim_scores + [dim_scores[0]],
        theta=dim_names + [dim_names[0]],
        fill="toself",
        fillcolor="rgba(0,102,204,0.2)",
        line=dict(color="#0066cc", width=2),
        name="Model Score",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[4.0] * len(dim_names) + [4.0],
        theta=dim_names + [dim_names[0]],
        mode="lines",
        line=dict(color="#198754", width=1, dash="dash"),
        name="Target (4.0)",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True, height=420,
        title="SR 11-7 Dimension Scores (Target ≥ 4.0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Dimension detail cards
    for d in sr117.dimensions:
        color = "#dc3545" if d.score < 2.5 else ("#fd7e14" if d.score < 3.5 else "#198754")
        with st.expander(f"{'🔴' if d.score<2.5 else ('🟡' if d.score<3.5 else '🟢')} {d.name}  —  Score: {d.score:.1f}/5.0"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Findings:**")
                for f in d.findings:
                    st.markdown(f"- {f}")
                if d.recommendations:
                    st.markdown("**Recommendations:**")
                    for r in d.recommendations:
                        st.markdown(f"- ➡️ {r}")
            with col2:
                st.markdown("**Evidence:**")
                for e in d.evidence:
                    st.code(e, language=None)
                bar = st.progress(d.score / 5.0)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — BIAS ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### Fairness & Bias Analysis")
    st.caption("Metrics: Disparate Impact Ratio (4/5 Rule), Statistical Parity Difference, Equal Opportunity Difference")

    bias_df = R["bias_df"]

    def color_risk(val):
        if val == "HIGH":   return "background-color:#f8d7da; color:#842029; font-weight:700"
        if val == "MEDIUM": return "background-color:#fff3cd; color:#856404; font-weight:700"
        return "background-color:#d1e7dd; color:#0f5132; font-weight:700"

    st.dataframe(
        bias_df.style.applymap(color_risk, subset=["Risk Level"]),
        use_container_width=True, hide_index=True
    )

    # Disparate Impact bar chart
    fig_bias = go.Figure()
    for r in bias:
        color = "#dc3545" if r.risk_level == "HIGH" else ("#fd7e14" if r.risk_level == "MEDIUM" else "#198754")
        fig_bias.add_trace(go.Bar(
            name=f"{r.attribute}: {r.group}",
            x=[f"{r.attribute}: {r.group} vs {r.reference_group}"],
            y=[r.disparate_impact_ratio],
            marker_color=color,
        ))
    fig_bias.add_hline(y=0.80, line_dash="dash", line_color="#dc3545",
                       annotation_text="4/5 Rule Threshold (0.80)", annotation_position="top right")
    fig_bias.update_layout(title="Disparate Impact Ratio by Group", yaxis_title="DIR", height=380, showlegend=False)
    st.plotly_chart(fig_bias, use_container_width=True)

    # Findings
    st.markdown("### 📝 Detailed Findings")
    for r in bias:
        emoji = "🔴" if r.risk_level == "HIGH" else ("🟡" if r.risk_level == "MEDIUM" else "🟢")
        with st.expander(f"{emoji} {r.attribute.title()} — Group: '{r.group}' vs '{r.reference_group}'"):
            for finding in r.findings:
                st.markdown(f"- {finding}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRIFT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### Model Drift Detection")
    st.markdown(drift.summary)

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Overall PSI",         f"{drift.overall_psi:.4f}")
    dc2.metric("Drifted Features",    drift.n_drifted_features)
    dc3.metric("Significant Drift",   drift.n_significant_features)
    dc4.metric("AUC Drop",            f"{drift.auc_drop:+.4f}", delta_color="inverse")

    # PSI bar chart
    drift_df = R["drift_df"]
    fig_psi = px.bar(
        drift_df.head(10), x="Feature", y="PSI",
        color="Severity",
        color_discrete_map={"SIGNIFICANT": "#dc3545", "MODERATE": "#fd7e14", "NONE": "#198754"},
        title="Population Stability Index (PSI) by Feature — Top 10",
    )
    fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#fd7e14", annotation_text="Moderate (0.10)")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#dc3545", annotation_text="Significant (0.25)")
    fig_psi.update_layout(height=380)
    st.plotly_chart(fig_psi, use_container_width=True)

    # Performance comparison
    fig_auc = go.Figure(go.Bar(
        x=["Baseline (Test Set)", "Production (Drifted)"],
        y=[drift.baseline_auc, drift.production_auc],
        marker_color=["#0066cc", "#dc3545" if drift.performance_degraded else "#198754"],
        text=[f"{drift.baseline_auc:.4f}", f"{drift.production_auc:.4f}"],
        textposition="outside",
    ))
    fig_auc.update_layout(title="AUC-ROC: Baseline vs Production", yaxis_range=[0, 1], height=320)
    st.plotly_chart(fig_auc, use_container_width=True)

    st.dataframe(drift_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### Model Explainability (SHAP)")
    st.caption("SHAP = SHapley Additive exPlanations — measures each feature's contribution to model predictions")

    # Global importance bar
    fig_shap = px.bar(
        shap_i.head(10), x="Mean |SHAP|", y="Feature",
        orientation="h", title="Top 10 Features — Mean |SHAP| Value",
        color="Mean |SHAP|", color_continuous_scale="Blues",
    )
    fig_shap.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig_shap, use_container_width=True)

    # Sklearn feature importance comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Features by SHAP**")
        st.dataframe(shap_i.head(10)[["Rank", "Feature", "Mean |SHAP|"]], hide_index=True)
    with col2:
        st.markdown("**Top Features by Model Importance**")
        st.dataframe(fi.head(10).reset_index(drop=True), hide_index=True)

    # SHAP distribution heatmap
    shap_df = R["shap_df"]
    top5    = shap_i.head(5)["Feature"].tolist()
    shap_sample = shap_df[top5].sample(min(200, len(shap_df)), random_state=42)

    fig_heat = px.imshow(
        shap_sample.T, aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="SHAP Value Heatmap — Top 5 Features (sample of 200 instances)",
        labels=dict(x="Sample Index", y="Feature", color="SHAP Value"),
    )
    fig_heat.update_layout(height=320)
    st.plotly_chart(fig_heat, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — MRM REPORT
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    ai_badge = "🤖 AI-Generated" if mrm.get("ai_generated") else "📋 Structured"
    st.markdown(f"### Model Risk Management Report  `{ai_badge}`")
    st.caption(f"Model: {mrm['model_name']}  |  Review Date: {mrm['review_date']}  |  Next Review: {mrm.get('next_review_date','TBD')}")

    # Verdict
    st.markdown(f"""
<div style='background:{bg_color}; border:2px solid {txt_color}; padding:1rem; border-radius:8px; margin-bottom:1rem;'>
<b style='color:{txt_color}; font-size:1.1rem;'>VERDICT: {verdict}</b><br>
SR 11-7 Score: <b>{mrm['sr117_score']:.2f}/5.00</b>  |  Risk Classification: <b>{mrm['sr117_tier']}</b>
</div>
""", unsafe_allow_html=True)

    # Executive Summary
    st.markdown("#### Executive Summary")
    st.markdown(mrm.get("executive_summary", ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔑 Key Findings")
        for i, finding in enumerate(mrm.get("key_findings", []), 1):
            st.markdown(f"{i}. {finding}")

    with col2:
        st.markdown("#### ➡️ Recommendations")
        for i, rec in enumerate(mrm.get("recommendations", []), 1):
            st.markdown(f"{i}. {rec}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚖️ Bias Assessment")
        st.markdown(mrm.get("bias_assessment", ""))
        st.markdown("#### 📡 Drift Assessment")
        st.markdown(mrm.get("drift_assessment", ""))
    with c2:
        st.markdown("#### 🔬 Explainability")
        st.markdown(mrm.get("explainability_assessment", ""))
        st.markdown("#### 📋 SR 11-7 Assessment")
        st.markdown(mrm.get("sr117_assessment", ""))

    # Download JSON report
    import json
    report_json = {k: v for k, v in mrm.items() if isinstance(v, (str, int, float, list, bool, type(None)))}
    st.download_button(
        "📥 Download Report (JSON)",
        data=json.dumps(report_json, indent=2),
        file_name=f"mrm_report_{mrm['review_date']}.json",
        mime="application/json",
    )

st.divider()
st.caption("MRM Auditor v1.0 | Built with SR 11-7 / SS1-23 | Powered by Claude + SHAP + Evidently")

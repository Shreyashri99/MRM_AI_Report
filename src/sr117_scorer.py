"""
Phase 5 — SR 11-7 Scorer
Scores the model against the Federal Reserve's SR 11-7 Model Risk Management
guidance (the gold standard for model risk in US banks).

Also maps to SS1/23 (PRA/Bank of England) for UK/EU coverage.

Six dimensions scored 1–5:
  1. Conceptual Soundness     — Is the model theory appropriate?
  2. Data Quality             — Are inputs valid, clean, representative?
  3. Performance Monitoring   — Is model performance tracked over time?
  4. Sensitivity Analysis     — How stable are outputs under input changes?
  5. Ongoing Validation       — Is there independent model validation?
  6. Documentation            — Is the model documented to regulatory standard?

Overall score → Risk Tier:
  4.0-5.0 = Tier 1 (Low Risk)
  3.0-3.9 = Tier 2 (Moderate Risk)
  2.0-2.9 = Tier 3 (Elevated Risk)
  1.0-1.9 = Tier 4 (High Risk — Restricted Use)
"""
import numpy as np
from dataclasses import dataclass, field
from config import SR117_WEIGHTS


@dataclass
class SR117Dimension:
    name:           str
    score:          float       # 1.0 – 5.0
    weight:         float
    weighted_score: float
    findings:       list[str]
    recommendations: list[str]
    evidence:       list[str]


@dataclass
class SR117Report:
    dimensions:     list[SR117Dimension]
    overall_score:  float
    risk_tier:      str
    tier_label:     str
    findings:       list[str]
    passed:         bool


def compute_sr117_score(
    baseline_metrics: dict,
    drift_report,
    bias_results: list,
    explainability_engine,
    has_documentation: bool = False,
    has_independent_validation: bool = False,
) -> SR117Report:
    """
    Scores the model across all SR 11-7 dimensions.
    Inputs are the outputs from previous phases.
    """
    print("\n📋  Running SR 11-7 Compliance Scoring...")
    dimensions = []

    # ── 1. Conceptual Soundness ────────────────────────────────────────────
    auc = baseline_metrics.get("auc_roc", 0)
    gi  = explainability_engine.global_importance()
    top_feature_shap = gi.iloc[0]["Mean |SHAP|"] if len(gi) > 0 else 0

    cs_score = 5.0
    cs_findings = []
    cs_recs     = []
    if auc < 0.65:
        cs_score -= 2.0; cs_findings.append("Model AUC below 0.65 — poor discriminatory power")
        cs_recs.append("Reconsider model architecture or feature engineering")
    elif auc < 0.75:
        cs_score -= 1.0; cs_findings.append("Model AUC between 0.65-0.75 — moderate performance")
    if top_feature_shap < 0.01:
        cs_score -= 0.5; cs_findings.append("Feature contributions are very low — model may not have learned meaningful patterns")
    if not cs_findings:
        cs_findings.append(f"Model AUC = {auc:.4f}. Discriminatory power is adequate.")

    dimensions.append(SR117Dimension(
        name="Conceptual Soundness", score=max(1.0, cs_score), weight=SR117_WEIGHTS["conceptual_soundness"],
        weighted_score=round(max(1.0, cs_score) * SR117_WEIGHTS["conceptual_soundness"], 4),
        findings=cs_findings, recommendations=cs_recs,
        evidence=[f"AUC-ROC: {auc:.4f}", f"Top SHAP feature: {gi.iloc[0]['Feature'] if len(gi)>0 else 'N/A'}"]
    ))

    # ── 2. Data Quality ────────────────────────────────────────────────────
    n_samples    = baseline_metrics.get("n_samples", 0)
    pos_rate     = baseline_metrics.get("positive_rate", 0)
    imbalance_ok = 0.1 <= pos_rate <= 0.9

    dq_score    = 5.0
    dq_findings = []
    dq_recs     = []
    if n_samples < 1000:
        dq_score -= 2.0; dq_findings.append(f"Test set too small ({n_samples:,} rows) for reliable assessment")
        dq_recs.append("Increase sample size to at least 10,000 records")
    if not imbalance_ok:
        dq_score -= 1.5; dq_findings.append(f"Significant class imbalance detected (positive rate: {pos_rate:.1%})")
        dq_recs.append("Apply SMOTE or class weighting; use F1/AUC as primary metrics")
    if drift_report.n_drifted_features > 0:
        dq_score -= 0.5; dq_findings.append(f"{drift_report.n_drifted_features} features show distribution shift vs baseline")
    if not dq_findings:
        dq_findings.append("Data quality checks passed. Sufficient sample size and balanced classes.")

    dimensions.append(SR117Dimension(
        name="Data Quality", score=max(1.0, dq_score), weight=SR117_WEIGHTS["data_quality"],
        weighted_score=round(max(1.0, dq_score) * SR117_WEIGHTS["data_quality"], 4),
        findings=dq_findings, recommendations=dq_recs,
        evidence=[f"Test samples: {n_samples:,}", f"Positive rate: {pos_rate:.1%}",
                  f"Drifted features: {drift_report.n_drifted_features}"]
    ))

    # ── 3. Performance Monitoring ──────────────────────────────────────────
    auc_drop = drift_report.auc_drop
    n_sig    = drift_report.n_significant_features

    pm_score    = 5.0
    pm_findings = []
    pm_recs     = []
    if drift_report.performance_degraded:
        pm_score -= 2.0
        pm_findings.append(f"AUC degraded by {auc_drop:.4f} on production data — exceeds {0.05:.0%} threshold")
        pm_recs.append("Trigger model recalibration. Set up automated AUC monitoring alerts.")
    if n_sig >= 3:
        pm_score -= 1.0
        pm_findings.append(f"{n_sig} features show significant PSI (>0.25) — input stability concern")
        pm_recs.append("Implement PSI dashboard and alert when any feature PSI > 0.20")
    if not pm_findings:
        pm_findings.append("Model performance is stable. No significant AUC degradation detected.")

    dimensions.append(SR117Dimension(
        name="Performance Monitoring", score=max(1.0, pm_score), weight=SR117_WEIGHTS["performance_monitoring"],
        weighted_score=round(max(1.0, pm_score) * SR117_WEIGHTS["performance_monitoring"], 4),
        findings=pm_findings, recommendations=pm_recs,
        evidence=[f"Baseline AUC: {drift_report.baseline_auc:.4f}",
                  f"Production AUC: {drift_report.production_auc:.4f}",
                  f"AUC drop: {auc_drop:+.4f}"]
    ))

    # ── 4. Sensitivity Analysis ────────────────────────────────────────────
    shap_df = explainability_engine.shap_dataframe()
    shap_std = shap_df.std().mean()
    concentration = gi.iloc[0]["Mean |SHAP|"] / gi["Mean |SHAP|"].sum() if gi["Mean |SHAP|"].sum() > 0 else 0

    sa_score    = 5.0
    sa_findings = []
    sa_recs     = []
    if concentration > 0.4:
        sa_score -= 1.5
        sa_findings.append(f"Top feature accounts for {concentration:.1%} of total SHAP — over-reliance on single driver")
        sa_recs.append("Investigate model dependence on top feature; consider regularisation")
    if shap_std < 0.001:
        sa_score -= 1.0
        sa_findings.append("Very low SHAP variance — model may be insensitive to input changes")
    if not sa_findings:
        sa_findings.append(f"Feature concentration healthy. Top feature accounts for {concentration:.1%} of SHAP mass.")

    dimensions.append(SR117Dimension(
        name="Sensitivity Analysis", score=max(1.0, sa_score), weight=SR117_WEIGHTS["sensitivity_analysis"],
        weighted_score=round(max(1.0, sa_score) * SR117_WEIGHTS["sensitivity_analysis"], 4),
        findings=sa_findings, recommendations=sa_recs,
        evidence=[f"Top feature SHAP concentration: {concentration:.1%}", f"Mean SHAP std: {shap_std:.5f}"]
    ))

    # ── 5. Ongoing Validation ──────────────────────────────────────────────
    high_bias = sum(1 for r in bias_results if r.risk_level == "HIGH")
    med_bias  = sum(1 for r in bias_results if r.risk_level == "MEDIUM")

    ov_score    = 5.0 if has_independent_validation else 2.5
    ov_findings = []
    ov_recs     = []
    if not has_independent_validation:
        ov_findings.append("No independent model validation documented — SR 11-7 requires separation of development and validation")
        ov_recs.append("Establish a Model Validation function independent of model development")
    if high_bias > 0:
        ov_score -= 1.5
        ov_findings.append(f"{high_bias} protected attribute group(s) show HIGH bias risk")
        ov_recs.append("Conduct full fair lending analysis; consult legal/compliance before deployment")
    if med_bias > 0:
        ov_score -= 0.5
        ov_findings.append(f"{med_bias} group(s) show MEDIUM bias risk")
    if not ov_findings:
        ov_findings.append("Validation process is adequate. No material bias issues detected.")

    dimensions.append(SR117Dimension(
        name="Ongoing Validation", score=max(1.0, ov_score), weight=SR117_WEIGHTS["ongoing_validation"],
        weighted_score=round(max(1.0, ov_score) * SR117_WEIGHTS["ongoing_validation"], 4),
        findings=ov_findings, recommendations=ov_recs,
        evidence=[f"High bias groups: {high_bias}", f"Medium bias groups: {med_bias}",
                  f"Independent validation: {'Yes' if has_independent_validation else 'No'}"]
    ))

    # ── 6. Documentation ──────────────────────────────────────────────────
    doc_score    = 5.0 if has_documentation else 1.5
    doc_findings = []
    doc_recs     = []
    if not has_documentation:
        doc_findings.append("No model documentation artefact found — required under SR 11-7 Section IV")
        doc_recs.append("Produce a Model Development Document (MDD) covering: purpose, methodology, data, limitations, validation plan")
    else:
        doc_findings.append("Model documentation is present.")

    dimensions.append(SR117Dimension(
        name="Documentation", score=max(1.0, doc_score), weight=SR117_WEIGHTS["documentation"],
        weighted_score=round(max(1.0, doc_score) * SR117_WEIGHTS["documentation"], 4),
        findings=doc_findings, recommendations=doc_recs,
        evidence=[f"Documentation present: {'Yes' if has_documentation else 'No'}"]
    ))

    # ── Aggregate ──────────────────────────────────────────────────────────
    overall = sum(d.weighted_score for d in dimensions) / sum(d.weight for d in dimensions)
    overall = round(overall, 3)

    if overall >= 4.0:
        risk_tier, tier_label = "Tier 1", "Low Risk — Approved for Use"
    elif overall >= 3.0:
        risk_tier, tier_label = "Tier 2", "Moderate Risk — Approved with Conditions"
    elif overall >= 2.0:
        risk_tier, tier_label = "Tier 3", "Elevated Risk — Restricted Use / Remediation Required"
    else:
        risk_tier, tier_label = "Tier 4", "High Risk — Use Prohibited Pending Remediation"

    all_findings = [f for d in dimensions for f in d.findings]
    passed       = overall >= 3.0

    print(f"\n    ┌─────────────────────────────────────────────┐")
    print(f"    │  SR 11-7 Overall Score : {overall:.2f} / 5.00         │")
    print(f"    │  Risk Classification   : {risk_tier} — {tier_label[:20]:<20}│")
    print(f"    └─────────────────────────────────────────────┘")
    for d in dimensions:
        bar = "█" * int(d.score) + "░" * (5 - int(d.score))
        print(f"    {d.name:<25} [{bar}] {d.score:.1f}")

    return SR117Report(
        dimensions=dimensions, overall_score=overall,
        risk_tier=risk_tier, tier_label=tier_label,
        findings=all_findings, passed=passed
    )

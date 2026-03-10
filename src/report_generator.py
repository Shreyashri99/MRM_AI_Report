"""
Phase 6 — Report Generator
Uses Claude API to generate a professional Model Risk Management Report
in the style of a bank's MRM/Model Validation team.

FIX: API key is read at call time (os.environ.get) not import time,
so the Streamlit sidebar input is always picked up correctly.
"""
import anthropic
import json
import os
import datetime
from config import CLAUDE_MODEL


def generate_mrm_report(
    baseline_metrics: dict,
    drift_report,
    bias_results: list,
    sr117_report,
    explainability_engine,
    model_name: str = "Credit Scoring Model v1.0",
) -> dict:
    """
    Calls Claude API to generate a full Model Risk Report.
    Reads API key fresh at call time so Streamlit sidebar input works.
    """
    # ── Read key at call time, not import time ─────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        print("⚠️  No Anthropic API key — returning structured report without AI narrative.")
        return _fallback_report(baseline_metrics, drift_report, bias_results, sr117_report, model_name)

    print("\n📝  Generating MRM Report via Claude API...")

    # ── Build context for Claude ───────────────────────────────────────────
    bias_summary = []
    for r in bias_results:
        bias_summary.append({
            "attribute":        r.attribute,
            "group":            r.group,
            "disparate_impact": r.disparate_impact_ratio,
            "stat_parity_diff": r.statistical_parity_diff,
            "risk_level":       r.risk_level,
        })

    sr117_dims = [
        {"dimension": d.name, "score": d.score, "findings": d.findings[:2]}
        for d in sr117_report.dimensions
    ]

    try:
        gi_top = explainability_engine.global_importance().head(5).to_dict("records")
    except Exception:
        gi_top = []

    context = {
        "model_name":   model_name,
        "review_date":  datetime.date.today().isoformat(),
        "performance":  baseline_metrics,
        "drift_summary": {
            "overall_psi":          drift_report.overall_psi,
            "drifted_features":     drift_report.n_drifted_features,
            "significant_features": drift_report.n_significant_features,
            "auc_drop":             drift_report.auc_drop,
            "production_auc":       drift_report.production_auc,
            "summary":              drift_report.summary,
        },
        "bias_results":     bias_summary,
        "sr117_score":      sr117_report.overall_score,
        "sr117_tier":       sr117_report.risk_tier,
        "sr117_tier_label": sr117_report.tier_label,
        "sr117_dimensions": sr117_dims,
        "top_features":     gi_top,
    }

    prompt = f"""You are a Senior Model Risk Analyst at a Tier 1 investment bank writing a formal
Model Risk Management (MRM) Validation Report. This report will be reviewed by the Chief Risk
Officer and presented to the Model Risk Committee.

Model Under Review: {model_name}
Review Date: {datetime.date.today().strftime('%d %B %Y')}

Audit Results:
{json.dumps(context, indent=2)}

Write a comprehensive, professional MRM report. Use formal banking language throughout.
Reference specific numerical findings (exact AUC values, PSI scores, DIR ratios).
Be direct about risks — do not soften findings.

Return a JSON object with EXACTLY these keys (no markdown, no extra keys, just valid JSON):
{{
  "executive_summary": "Write 3 full paragraphs. Para 1: model purpose and overall assessment. Para 2: the most critical risk findings with specific numbers. Para 3: overall recommendation for the risk committee.",
  "key_findings": [
    "Finding 1 with specific metric",
    "Finding 2 with specific metric",
    "Finding 3 with specific metric",
    "Finding 4 with specific metric",
    "Finding 5 with specific metric"
  ],
  "bias_assessment": "Write 2 full paragraphs. Reference exact Disparate Impact Ratios, which groups failed the 4/5 Rule, Statistical Parity Differences, and the legal/regulatory implications under ECOA and Regulation B.",
  "drift_assessment": "Write 2 full paragraphs. Name the specific features with high PSI, state the exact AUC degradation, and assess whether the model remains fit for purpose under current production conditions.",
  "explainability_assessment": "Write 1 full paragraph. Name the top 3 model drivers by SHAP value, assess whether these drivers are conceptually sound for a credit decision, and comment on whether the model could be explained to a regulator.",
  "sr117_assessment": "Write 2 full paragraphs. Map each of the 6 SR 11-7 dimensions to their scores. Explicitly call out the lowest-scoring dimensions and what they mean for regulatory compliance.",
  "recommendations": [
    "Specific recommendation 1 with owner and timeline",
    "Specific recommendation 2 with owner and timeline",
    "Specific recommendation 3 with owner and timeline",
    "Specific recommendation 4 with owner and timeline",
    "Specific recommendation 5 with owner and timeline"
  ],
  "model_verdict": "APPROVED WITH CONDITIONS",
  "verdict_rationale": "Write 1 full paragraph justifying the verdict with reference to specific scores. State exactly what conditions must be met before unrestricted approval.",
  "next_review_date": "{(datetime.date.today().replace(year=datetime.date.today().year + 1)).isoformat()}"
}}

Return ONLY the JSON object. No markdown fences. No preamble. No explanation."""

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = message.content[0].text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0].strip()

    try:
        report_data = json.loads(raw_text)
        report_data["model_name"]   = model_name
        report_data["review_date"]  = datetime.date.today().isoformat()
        report_data["sr117_score"]  = sr117_report.overall_score
        report_data["sr117_tier"]   = sr117_report.risk_tier
        report_data["ai_generated"] = True
        print("✅  MRM Report generated successfully via Claude API.")
        return report_data
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parse error: {e}")
        print(f"    Raw response (first 300 chars): {raw_text[:300]}")
        print("    Falling back to structured report.")
        return _fallback_report(baseline_metrics, drift_report, bias_results, sr117_report, model_name)


def _fallback_report(baseline_metrics, drift_report, bias_results, sr117_report, model_name) -> dict:
    """Structured report without AI narrative — used when no API key is provided."""
    high_bias = [r for r in bias_results if r.risk_level == "HIGH"]
    verdict   = "APPROVED" if sr117_report.passed and not high_bias else "APPROVED WITH CONDITIONS"

    return {
        "model_name":    model_name,
        "review_date":   datetime.date.today().isoformat(),
        "sr117_score":   sr117_report.overall_score,
        "sr117_tier":    sr117_report.risk_tier,
        "ai_generated":  False,
        "executive_summary": (
            f"[Structured Report — Add Anthropic API key for full AI narrative]\n\n"
            f"This report presents the model risk assessment for {model_name}, conducted in "
            f"accordance with SR 11-7 / SS1-23 Model Risk Management guidance. "
            f"The model achieved an AUC-ROC of {baseline_metrics.get('auc_roc', 0):.4f} on the holdout "
            f"test set. The overall SR 11-7 compliance score is {sr117_report.overall_score:.2f}/5.00, "
            f"classifying the model as {sr117_report.risk_tier} ({sr117_report.tier_label}). "
            f"{len(high_bias)} protected attribute group(s) exhibit HIGH bias risk, requiring "
            f"immediate remediation prior to production deployment."
        ),
        "key_findings": [
            f.replace("⚠️  ", "").replace("✅  ", "").replace("ℹ️  ", "")
            for f in sr117_report.findings[:5]
        ],
        "bias_assessment": (
            f"{len(high_bias)} group(s) show HIGH bias risk. "
            f"Disparate impact analysis completed for gender and race attributes. "
            f"Add API key for full AI-written narrative."
        ),
        "drift_assessment": (
            f"Overall PSI = {drift_report.overall_psi:.4f}. "
            f"{drift_report.n_drifted_features} features show drift. "
            f"AUC drop = {drift_report.auc_drop:+.4f}. "
            f"Add API key for full AI-written narrative."
        ),
        "explainability_assessment": "SHAP analysis completed. Add API key for full AI-written narrative.",
        "sr117_assessment": (
            f"SR 11-7 overall score: {sr117_report.overall_score:.2f}/5. "
            f"Classification: {sr117_report.tier_label}. "
            f"Add API key for full AI-written narrative."
        ),
        "recommendations": [
            d.recommendations[0] for d in sr117_report.dimensions if d.recommendations
        ],
        "model_verdict":     verdict,
        "verdict_rationale": (
            f"Based on SR 11-7 score of {sr117_report.overall_score:.2f}/5 and "
            f"{len(high_bias)} high-risk bias finding(s). Add API key for full rationale."
        ),
        "next_review_date": (
            datetime.date.today().replace(year=datetime.date.today().year + 1)
        ).isoformat(),
    }

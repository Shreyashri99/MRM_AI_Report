"""
Phase 2 — Bias Detector
Computes fairness metrics across protected attributes.
Implements the key metrics used in SR 11-7 / fair lending:
  • Disparate Impact Ratio (4/5 Rule — EEOC standard)
  • Statistical Parity Difference
  • Equal Opportunity Difference (True Positive Rate parity)
  • Predictive Parity (Precision parity)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config import (
    BIAS_THRESHOLD_DISPARATE_IMPACT,
    BIAS_THRESHOLD_STAT_PARITY_DIFF,
    PROTECTED_ATTRIBUTES,
    POSITIVE_LABEL,
)


@dataclass
class BiasResult:
    attribute:                  str
    group:                      str
    reference_group:            str
    group_positive_rate:        float
    reference_positive_rate:    float
    disparate_impact_ratio:     float
    statistical_parity_diff:    float
    equal_opportunity_diff:     float
    precision_parity_diff:      float
    bias_detected:              bool
    risk_level:                 str     # LOW / MEDIUM / HIGH
    findings:                   list    = field(default_factory=list)


def compute_bias_metrics(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_raw: pd.DataFrame,
) -> list[BiasResult]:
    """
    Runs bias analysis for all protected attributes.
    Returns a list of BiasResult objects.
    """
    results = []
    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]

    # Align raw demo columns with test set index
    demo_df = df_raw.loc[X_test.index].copy() if len(df_raw) > len(X_test) else df_raw.copy()
    demo_df = demo_df.reset_index(drop=True)
    y_test_r  = y_test.reset_index(drop=True)
    y_pred_s  = pd.Series(y_pred, name="y_pred")

    combined = demo_df.copy()
    combined["y_true"]  = y_test_r.values
    combined["y_pred"]  = y_pred
    combined["y_prob"]  = y_prob

    for attr in PROTECTED_ATTRIBUTES:
        if attr not in combined.columns:
            continue

        groups = combined[attr].unique()
        # Pick reference group: most frequent
        ref_group = combined[attr].value_counts().idxmax()

        ref_data  = combined[combined[attr] == ref_group]
        ref_pos_rate   = (ref_data["y_pred"] == POSITIVE_LABEL).mean()
        ref_tpr        = _true_positive_rate(ref_data)
        ref_precision  = _precision(ref_data)

        for group in groups:
            if group == ref_group:
                continue

            grp_data      = combined[combined[attr] == group]
            grp_pos_rate  = (grp_data["y_pred"] == POSITIVE_LABEL).mean()
            grp_tpr       = _true_positive_rate(grp_data)
            grp_precision = _precision(grp_data)

            # ── Core metrics ───────────────────────────────────────────────
            dir_val  = grp_pos_rate / ref_pos_rate if ref_pos_rate > 0 else 0.0
            spd      = grp_pos_rate - ref_pos_rate
            eod      = grp_tpr - ref_tpr
            ppd      = grp_precision - ref_precision

            # ── Risk classification ────────────────────────────────────────
            findings   = []
            bias_flags = 0

            if dir_val < BIAS_THRESHOLD_DISPARATE_IMPACT:
                bias_flags += 2
                findings.append(
                    f"⚠️  Disparate Impact Ratio = {dir_val:.3f} (threshold: ≥{BIAS_THRESHOLD_DISPARATE_IMPACT}) "
                    f"— violates the 4/5 Rule. Group '{group}' receives positive outcomes "
                    f"at only {dir_val*100:.1f}% the rate of '{ref_group}'."
                )
            if abs(spd) > BIAS_THRESHOLD_STAT_PARITY_DIFF:
                bias_flags += 1
                findings.append(
                    f"⚠️  Statistical Parity Difference = {spd:+.3f} — "
                    f"group '{group}' positive rate ({grp_pos_rate:.1%}) differs "
                    f"from reference ({ref_pos_rate:.1%}) by >{BIAS_THRESHOLD_STAT_PARITY_DIFF*100:.0f}pp."
                )
            if abs(eod) > 0.05:
                bias_flags += 1
                findings.append(
                    f"ℹ️  Equal Opportunity Difference = {eod:+.3f} — "
                    f"True positive rates diverge between groups."
                )

            if bias_flags >= 2:
                risk_level = "HIGH"
            elif bias_flags == 1:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                findings.append(f"✅  No significant bias detected for group '{group}'.")

            results.append(BiasResult(
                attribute              = attr,
                group                  = group,
                reference_group        = ref_group,
                group_positive_rate    = round(grp_pos_rate, 4),
                reference_positive_rate= round(ref_pos_rate, 4),
                disparate_impact_ratio = round(dir_val, 4),
                statistical_parity_diff= round(spd, 4),
                equal_opportunity_diff = round(eod, 4),
                precision_parity_diff  = round(ppd, 4),
                bias_detected          = bias_flags > 0,
                risk_level             = risk_level,
                findings               = findings,
            ))

    _print_bias_summary(results)
    return results


def _true_positive_rate(df: pd.DataFrame) -> float:
    positives = df[df["y_true"] == POSITIVE_LABEL]
    if len(positives) == 0:
        return 0.0
    return (positives["y_pred"] == POSITIVE_LABEL).mean()


def _precision(df: pd.DataFrame) -> float:
    predicted_pos = df[df["y_pred"] == POSITIVE_LABEL]
    if len(predicted_pos) == 0:
        return 0.0
    return (predicted_pos["y_true"] == POSITIVE_LABEL).mean()


def _print_bias_summary(results: list[BiasResult]):
    print("\n🔍  Bias Analysis Results")
    print(f"    {'Attribute':<12} {'Group':<25} {'DIR':>6} {'SPD':>7} {'Risk':<8}")
    print("    " + "-" * 60)
    for r in results:
        emoji = "🔴" if r.risk_level == "HIGH" else ("🟡" if r.risk_level == "MEDIUM" else "🟢")
        print(f"    {r.attribute:<12} {r.group:<25} {r.disparate_impact_ratio:>6.3f} "
              f"{r.statistical_parity_diff:>+7.3f}  {emoji} {r.risk_level}")


def bias_summary_dataframe(results: list[BiasResult]) -> pd.DataFrame:
    """Converts bias results to a display-ready DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "Protected Attribute": r.attribute,
            "Assessed Group":      r.group,
            "Reference Group":     r.reference_group,
            "Group Positive Rate": f"{r.group_positive_rate:.1%}",
            "Ref Positive Rate":   f"{r.reference_positive_rate:.1%}",
            "Disparate Impact":    f"{r.disparate_impact_ratio:.3f}",
            "Stat Parity Diff":    f"{r.statistical_parity_diff:+.3f}",
            "Equal Opp Diff":      f"{r.equal_opportunity_diff:+.3f}",
            "Risk Level":          r.risk_level,
        })
    return pd.DataFrame(rows)

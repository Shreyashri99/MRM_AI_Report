"""
Phase 3 — Drift Simulator
Detects data drift between training/baseline and production distributions.

Metrics implemented:
  • PSI  — Population Stability Index (industry standard in banking)
  • KS   — Kolmogorov-Smirnov test (distribution shift per feature)
  • Performance degradation tracking (AUC drop)

PSI Interpretation (Basel / SS1-23 standard):
  PSI < 0.10  → Negligible (no action needed)
  PSI 0.10-0.25 → Moderate (investigate)
  PSI > 0.25  → Significant (model may need refit/replacement)
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score
from config import DRIFT_PSI_LOW, DRIFT_PSI_MEDIUM, PERFORMANCE_DROP_THRESHOLD


@dataclass
class DriftResult:
    feature:        str
    psi:            float
    ks_statistic:   float
    ks_p_value:     float
    drift_detected: bool
    severity:       str         # NONE / MODERATE / SIGNIFICANT
    mean_shift:     float
    std_shift:      float


@dataclass
class DriftReport:
    feature_results:        list[DriftResult]
    overall_psi:            float
    n_drifted_features:     int
    n_significant_features: int
    baseline_auc:           float
    production_auc:         float
    auc_drop:               float
    performance_degraded:   bool
    summary:                str


def compute_psi(baseline: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    def _safe_pct(arr, edges):
        counts, _ = np.histogram(arr, bins=edges)
        pct       = counts / len(arr)
        return np.where(pct == 0, 1e-6, pct)

    # Use baseline distribution to define buckets
    min_val = min(baseline.min(), production.min())
    max_val = max(baseline.max(), production.max())
    edges   = np.linspace(min_val, max_val, buckets + 1)

    expected = _safe_pct(baseline, edges)
    actual   = _safe_pct(production, edges)

    psi = np.sum((actual - expected) * np.log(actual / expected))
    return round(float(psi), 5)


def analyse_drift(
    X_baseline:  pd.DataFrame,
    X_production: pd.DataFrame,
    pipeline,
    y_baseline:   pd.Series,
    y_production: pd.Series,
) -> DriftReport:
    """
    Full drift analysis comparing baseline (test) vs production (drifted) data.
    """
    print("\n📡  Running Drift Analysis...")
    feature_results = []
    numeric_cols = X_baseline.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        base_arr = X_baseline[col].values
        prod_arr = X_production[col].values

        psi           = compute_psi(base_arr, prod_arr)
        ks_stat, ks_p = stats.ks_2samp(base_arr, prod_arr)
        mean_shift    = prod_arr.mean() - base_arr.mean()
        std_shift     = prod_arr.std()  - base_arr.std()

        if psi >= DRIFT_PSI_MEDIUM:
            severity       = "SIGNIFICANT"
            drift_detected = True
        elif psi >= DRIFT_PSI_LOW:
            severity       = "MODERATE"
            drift_detected = True
        else:
            severity       = "NONE"
            drift_detected = False

        feature_results.append(DriftResult(
            feature        = col,
            psi            = psi,
            ks_statistic   = round(float(ks_stat), 4),
            ks_p_value     = round(float(ks_p), 6),
            drift_detected = drift_detected,
            severity       = severity,
            mean_shift     = round(float(mean_shift), 4),
            std_shift      = round(float(std_shift), 4),
        ))

    # ── Performance degradation ────────────────────────────────────────────
    base_prob   = pipeline.predict_proba(X_baseline)[:, 1]
    prod_prob   = pipeline.predict_proba(X_production)[:, 1]
    baseline_auc  = round(roc_auc_score(y_baseline, base_prob), 4)
    production_auc = round(roc_auc_score(y_production, prod_prob), 4)
    auc_drop      = round(baseline_auc - production_auc, 4)

    n_drifted     = sum(1 for r in feature_results if r.drift_detected)
    n_significant = sum(1 for r in feature_results if r.severity == "SIGNIFICANT")

    overall_psi   = np.mean([r.psi for r in feature_results])

    if n_significant >= 3 or auc_drop > PERFORMANCE_DROP_THRESHOLD:
        summary = "🔴  HIGH RISK — Significant drift detected. Model should be reviewed immediately."
    elif n_drifted > 0 or auc_drop > 0:
        summary = "🟡  MODERATE RISK — Drift detected in some features. Increase monitoring frequency."
    else:
        summary = "🟢  LOW RISK — No significant drift detected. Model is stable."

    report = DriftReport(
        feature_results        = sorted(feature_results, key=lambda r: r.psi, reverse=True),
        overall_psi            = round(float(overall_psi), 5),
        n_drifted_features     = n_drifted,
        n_significant_features = n_significant,
        baseline_auc           = baseline_auc,
        production_auc         = production_auc,
        auc_drop               = auc_drop,
        performance_degraded   = auc_drop > PERFORMANCE_DROP_THRESHOLD,
        summary                = summary,
    )

    _print_drift_summary(report)
    return report


def _print_drift_summary(r: DriftReport):
    print(f"\n    Overall PSI: {r.overall_psi:.4f}  |  Drifted features: {r.n_drifted_features}  |  Significant: {r.n_significant_features}")
    print(f"    Baseline AUC: {r.baseline_auc:.4f}  →  Production AUC: {r.production_auc:.4f}  (Drop: {r.auc_drop:+.4f})")
    print(f"    {r.summary}")


def drift_summary_dataframe(report: DriftReport) -> pd.DataFrame:
    rows = []
    for r in report.feature_results:
        rows.append({
            "Feature":        r.feature,
            "PSI":            f"{r.psi:.4f}",
            "KS Statistic":   f"{r.ks_statistic:.4f}",
            "KS p-value":     f"{r.ks_p_value:.4f}",
            "Mean Shift":     f"{r.mean_shift:+.4f}",
            "Severity":       r.severity,
            "Drift Detected": "Yes" if r.drift_detected else "No",
        })
    return pd.DataFrame(rows)

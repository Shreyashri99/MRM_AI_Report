"""
MRM Auditor — Main Pipeline Orchestrator
Runs all 6 phases end-to-end and caches results for the dashboard.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_adult_dataset, simulate_production_drift
from src.model_trainer import train_model, evaluate_model, get_feature_importance
from src.bias_detector import compute_bias_metrics, bias_summary_dataframe
from src.drift_simulator import analyse_drift, drift_summary_dataframe
from src.explainability_engine import ExplainabilityEngine
from src.sr117_scorer import compute_sr117_score
from src.report_generator import generate_mrm_report
import pickle, os, time


CACHE_PATH = os.path.join(os.path.dirname(__file__), "reports", "audit_cache.pkl")


def run_full_audit(
    model_type: str = "gradient_boosting",
    drift_intensity: float = 0.3,
    use_cache: bool = False,
    api_key: str = "",
) -> dict:
    """
    Runs the complete MRM audit pipeline.
    Returns a results dict consumed by the Streamlit dashboard.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        print("📦  Loading cached audit results...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    results = {}
    t0 = time.time()

    # ── Phase 1: Load data & train model ──────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 1 — Data Loading & Model Training")
    print("="*60)
    X_train, X_test, y_train, y_test, df_raw = load_adult_dataset()
    pipeline = train_model(X_train, y_train, model_type=model_type)
    baseline_metrics = evaluate_model(pipeline, X_test, y_test, "Holdout Test Set")
    feature_importance = get_feature_importance(pipeline, X_train.columns.tolist())

    results["X_train"]           = X_train
    results["X_test"]            = X_test
    results["y_train"]           = y_train
    results["y_test"]            = y_test
    results["df_raw"]            = df_raw
    results["pipeline"]          = pipeline
    results["baseline_metrics"]  = baseline_metrics
    results["feature_importance"] = feature_importance

    # ── Phase 2: Bias Detection ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 2 — Bias & Fairness Analysis")
    print("="*60)
    bias_results = compute_bias_metrics(pipeline, X_test, y_test, df_raw)
    results["bias_results"]      = bias_results
    results["bias_df"]           = bias_summary_dataframe(bias_results)

    # ── Phase 3: Drift Simulation ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 3 — Model Drift Detection")
    print("="*60)
    X_drift     = simulate_production_drift(X_test, drift_intensity=drift_intensity)
    drift_report = analyse_drift(X_test, X_drift, pipeline, y_test, y_test)
    results["X_drift"]           = X_drift
    results["drift_report"]      = drift_report
    results["drift_df"]          = drift_summary_dataframe(drift_report)

    # ── Phase 4: Explainability ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 4 — Explainability (SHAP)")
    print("="*60)
    engine = ExplainabilityEngine(pipeline, X_train, model_type)
    engine.fit(X_test, max_samples=300)
    results["explainability"]    = engine
    results["shap_importance"]   = engine.global_importance()
    results["shap_df"]           = engine.shap_dataframe()

    # ── Phase 5: SR 11-7 Scoring ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 5 — SR 11-7 Compliance Scoring")
    print("="*60)
    sr117 = compute_sr117_score(
        baseline_metrics, drift_report, bias_results, engine,
        has_documentation=False, has_independent_validation=False
    )
    results["sr117"] = sr117

    # ── Phase 6: Report Generation ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  PHASE 6 — MRM Report Generation")
    print("="*60)
    mrm_report = generate_mrm_report(
        baseline_metrics, drift_report, bias_results, sr117, engine,
        model_name="Credit Scoring Model v1.0"
    )
    results["mrm_report"] = mrm_report

    elapsed = round(time.time() - t0, 1)
    print(f"\n✅  Full audit completed in {elapsed}s")
    print(f"    Model Verdict: {mrm_report.get('model_verdict', 'N/A')}")

    # Cache results (excluding large objects for safety)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(results, f)
        print(f"    Results cached to {CACHE_PATH}")
    except Exception as e:
        print(f"    (Cache save skipped: {e})")

    return results


if __name__ == "__main__":
    results = run_full_audit(use_cache=False)
    print("\n🎯  All phases complete. Launch dashboard with: streamlit run app.py")

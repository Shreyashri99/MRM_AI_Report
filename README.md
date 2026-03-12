# 🏦 AI Model Risk Governance Engine
### SR 11-7 / SS1-23 Compliant Automated MRM Auditor

> **Built for:** Technology Risk professionals with banking/Big 4 backgrounds  
> **Regulatory Coverage:** Federal Reserve SR 11-7, PRA SS1/23, Basel principles  
> **Dataset:** UCI Adult Income (real credit-adjacent data with demographic attributes)

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MRM AUDITOR PIPELINE                             │
│                                                                     │
│  PHASE 1          PHASE 2         PHASE 3         PHASE 4          │
│  ─────────        ────────        ────────        ────────          │
│  Data Load   →   Bias         →   Drift       →   SHAP             │
│  + Train         Detection        Simulation      Explainability   │
│                  (DIR, SPD,       (PSI, KS,       (Global +        │
│                  EOD)             AUC drop)        Single)          │
│                                                                     │
│                      ↓                                              │
│  PHASE 5          PHASE 6         PHASE 7                          │
│  ─────────        ────────        ────────                          │
│  SR 11-7      →   Claude AI   →   Streamlit                        │
│  Scoring          Report Gen      Dashboard                        │
│  (6 dims)         (MRM format)    (Interactive)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mrm_auditor/
│
├── app.py                          # 🖥️  Streamlit Dashboard (Phase 7)
├── pipeline.py                     # 🔄  Full audit orchestrator
├── config.py                       # ⚙️  Thresholds, weights, settings
├── requirements.txt                # 📦  Dependencies
│
├── src/
│   ├── data_loader.py              # Phase 1: UCI Adult Income + drift simulation
│   ├── model_trainer.py            # Phase 1: GBM / Logistic Regression training
│   ├── bias_detector.py            # Phase 2: DIR, SPD, EOD, Precision Parity
│   ├── drift_simulator.py          # Phase 3: PSI, KS test, AUC degradation
│   ├── explainability_engine.py    # Phase 4: SHAP global + local explanations
│   ├── sr117_scorer.py             # Phase 5: 6-dimension SR 11-7 scoring
│   └── report_generator.py         # Phase 6: Claude API MRM report
│
└── reports/                        # Output directory for reports & cache
```

---

## 🚀 Quick Start

### 1. Clone / Download
```bash
git clone <your-repo>
cd mrm_auditor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3a. Run Pipeline Only (CLI)
```bash
python pipeline.py
```
This runs all 6 phases and prints results to terminal. Takes ~60-90s.

### 3b. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## ⚙️ Configuration

Edit `config.py` to change regulatory thresholds:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `BIAS_THRESHOLD_DISPARATE_IMPACT` | 0.80 | 4/5 Rule (EEOC standard) |
| `BIAS_THRESHOLD_STAT_PARITY_DIFF` | 0.10 | >10pp difference = flag |
| `DRIFT_PSI_LOW` | 0.10 | PSI threshold: moderate drift |
| `DRIFT_PSI_MEDIUM` | 0.25 | PSI threshold: significant drift |
| `PERFORMANCE_DROP_THRESHOLD` | 0.05 | >5% AUC drop = performance degraded |

SR 11-7 dimension weights (must sum to 1.0):
```python
SR117_WEIGHTS = {
    "conceptual_soundness":   0.25,
    "data_quality":           0.20,
    "performance_monitoring": 0.20,
    "sensitivity_analysis":   0.15,
    "ongoing_validation":     0.10,
    "documentation":          0.10,
}
```

---

## 🤖 AI-Powered Report (Claude API)

Add your Anthropic API key to get Claude to write the MRM narrative:

**Option A** — Environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python pipeline.py
```

**Option B** — .env file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

**Option C** — Dashboard:
Enter in the sidebar text field before running audit.

Without an API key, the tool still produces a full structured report — just without the AI narrative prose.

---

## 📊 What Each Phase Produces

### Phase 1: Data & Model
- Loads 45,222 records from UCI Adult Income dataset
- Trains a Gradient Boosting Classifier (mimics a credit scoring model)
- Produces holdout test metrics: AUC, Accuracy, Precision, Recall, F1

### Phase 2: Bias Analysis
Computes per protected attribute (gender, race):
- **Disparate Impact Ratio** (DIR) — the "4/5 Rule" from EEOC
- **Statistical Parity Difference** (SPD) — raw positive rate gap
- **Equal Opportunity Difference** (EOD) — true positive rate gap
- **Risk classification**: HIGH / MEDIUM / LOW per group

### Phase 3: Drift Detection
- Simulates production data by shifting feature distributions
- Computes **PSI** (Population Stability Index) per feature
- Runs **KS test** for distributional significance
- Measures **AUC degradation** on drifted data

### Phase 4: Explainability (SHAP)
- Fits TreeExplainer on 300-sample subset
- Produces global feature importance (mean |SHAP|)
- Generates SHAP heatmap across instances
- Measures feature concentration risk

### Phase 5: SR 11-7 Scoring
Six dimensions, each scored 1.0–5.0:
1. Conceptual Soundness (25%)
2. Data Quality (20%)
3. Performance Monitoring (20%)
4. Sensitivity Analysis (15%)
5. Ongoing Validation (10%)
6. Documentation (10%)

**Risk Tiers:**
- ≥4.0 → Tier 1 (Low Risk — Approved)
- 3.0–3.9 → Tier 2 (Moderate — Approved with Conditions)
- 2.0–2.9 → Tier 3 (Elevated — Restricted Use)
- <2.0 → Tier 4 (High Risk — Not Approved)

### Phase 6: MRM Report
Claude generates a professional report with:
- Executive Summary
- Key Findings
- Bias / Drift / Explainability Assessments
- SR 11-7 Assessment
- Recommendations
- Model Verdict (APPROVED / APPROVED WITH CONDITIONS / etc.)


## 📚 Regulatory References

- **SR 11-7** — Federal Reserve / OCC Model Risk Management Guidance (2011)
- **SS1/23** — PRA / Bank of England Model Risk Management (2023)
- **ECOA / Regulation B** — Equal Credit Opportunity Act (disparate impact)
- **Basel III** — Model risk as operational risk category
- **EEOC 4/5 Rule** — Disparate Impact threshold standard


## 👤 Author Notes

Built to demonstrate real-world application of AI to Model Risk Management.
This project directly addresses problems faced daily by MRM/Model Validation
teams at banks — bias reviews, model monitoring, SR 11-7 documentation gaps.

The scoring logic, thresholds, and regulatory references are grounded in
actual banking MRM practice, not just academic definitions.

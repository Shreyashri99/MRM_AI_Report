"""
MRM Auditor — Global Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"

# ── SR 11-7 Scoring Weights ────────────────────────────────────────────────
SR117_WEIGHTS = {
    "conceptual_soundness":   0.25,
    "data_quality":           0.20,
    "performance_monitoring": 0.20,
    "sensitivity_analysis":   0.15,
    "ongoing_validation":     0.10,
    "documentation":          0.10,
}

# ── Risk Thresholds ────────────────────────────────────────────────────────
BIAS_THRESHOLD_DISPARATE_IMPACT = 0.80   # Below this = High Risk (4/5 Rule)
BIAS_THRESHOLD_STAT_PARITY_DIFF = 0.10   # Above this = Elevated Risk
DRIFT_PSI_LOW    = 0.10   # PSI < 0.1  → Negligible drift
DRIFT_PSI_MEDIUM = 0.25   # PSI < 0.25 → Moderate drift
PERFORMANCE_DROP_THRESHOLD = 0.05        # >5% AUC drop = flag

# ── Demographic Columns (Adult Income Dataset) ─────────────────────────────
PROTECTED_ATTRIBUTES = ["gender", "race"]
LABEL_COLUMN         = "income"
POSITIVE_LABEL       = 1    # ">50K"

# ── Report Output ──────────────────────────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

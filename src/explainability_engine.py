"""
Phase 4 — Explainability Engine
Generates SHAP-based explanations for the model.

Why this matters for SR 11-7:
  "Model developers must provide sufficient documentation on the model's
   conceptual soundness, including the model's assumptions and limitations."

Outputs:
  • Global feature importance (mean |SHAP|)
  • SHAP summary data for plotting
  • Individual prediction explanation for a sample case
"""
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")


class ExplainabilityEngine:
    def __init__(self, pipeline, X_train: pd.DataFrame, model_type: str = "gradient_boosting"):
        self.pipeline   = pipeline
        self.X_train    = X_train
        self.model_type = model_type
        self.explainer  = None
        self.shap_values = None
        self._fitted     = False

    def fit(self, X_sample: pd.DataFrame = None, max_samples: int = 500):
        """Fit the SHAP explainer. Uses a sample for speed."""
        print("\n🔬  Fitting SHAP explainer...")
        clf = self.pipeline.named_steps["clf"]

        # Use a background sample for TreeExplainer
        background = self.X_train.sample(
            min(100, len(self.X_train)), random_state=42
        )
        X_explain = (X_sample if X_sample is not None else self.X_train).sample(
            min(max_samples, len(self.X_train)), random_state=42
        )

        try:
            # TreeExplainer is fast for GBM models
            self.explainer  = shap.TreeExplainer(clf, background)
            self.shap_values = self.explainer.shap_values(X_explain)
            self._X_explained = X_explain
        except Exception:
            # Fallback to KernelExplainer
            def predict_fn(x):
                return self.pipeline.predict_proba(
                    pd.DataFrame(x, columns=self.X_train.columns)
                )[:, 1]
            self.explainer  = shap.KernelExplainer(predict_fn, background)
            self.shap_values = self.explainer.shap_values(X_explain)
            self._X_explained = X_explain

        self._fitted = True
        print("✅  SHAP explainer fitted.")

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP values — global feature importance."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")

        sv = self.shap_values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]

        mean_shap = np.abs(sv).mean(axis=0)
        df = pd.DataFrame({
            "Feature":    self._X_explained.columns.tolist(),
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["Mean |SHAP|"] = df["Mean |SHAP|"].round(5)
        return df

    def shap_dataframe(self) -> pd.DataFrame:
        """Return raw SHAP values as a DataFrame."""
        sv = self.shap_values
        if isinstance(sv, list):
            sv = sv[1] if len(sv) == 2 else sv[0]
        return pd.DataFrame(sv, columns=self._X_explained.columns)

    def explain_single(self, X_instance: pd.Series) -> dict:
        """Explain a single prediction."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first")
        clf = self.pipeline.named_steps["clf"]
        try:
            exp   = shap.TreeExplainer(clf)
            sv    = exp.shap_values(X_instance.values.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else sv[0]
            sv = sv[0]
        except Exception:
            sv = np.zeros(len(X_instance))

        result = {
            "features":   X_instance.index.tolist(),
            "values":     X_instance.values.tolist(),
            "shap_values": sv.tolist(),
            "prediction":  int(self.pipeline.predict(X_instance.values.reshape(1, -1))[0]),
            "probability": float(self.pipeline.predict_proba(X_instance.values.reshape(1, -1))[0, 1]),
        }
        return result

    def top_features_narrative(self, n: int = 5) -> str:
        """Human-readable narrative of top features."""
        gi   = self.global_importance().head(n)
        lines = [f"The model's top {n} predictive features are:"]
        for _, row in gi.iterrows():
            lines.append(
                f"  {int(row['Rank'])}. {row['Feature']} (mean |SHAP| = {row['Mean |SHAP|']:.4f})"
            )
        return "\n".join(lines)

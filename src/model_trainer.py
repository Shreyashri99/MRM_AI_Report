"""
Phase 1 — Model Trainer
Trains an XGBoost classifier (mimics a real bank credit-scoring model).
Returns the trained model + evaluation metrics.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def train_model(X_train, y_train, model_type: str = "gradient_boosting"):
    """Train a classification model. Returns fitted pipeline."""
    print(f"\n🤖  Training {model_type} model...")

    if model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=4,
            learning_rate=0.1, random_state=42
        )
        pipeline = Pipeline([("clf", clf)])

    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline.fit(X_train, y_train)
    print("✅  Model trained successfully.")
    return pipeline


def evaluate_model(pipeline, X_test, y_test, dataset_label: str = "Test") -> dict:
    """Full evaluation — returns metrics dict."""
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset":   dataset_label,
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_samples": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }

    print(f"\n📊  [{dataset_label}] Performance Metrics")
    print(f"    AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1 Score  : {metrics['f1_score']:.4f}")

    return metrics


def get_feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from the model."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df

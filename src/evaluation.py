import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


def evaluate_model(y_true, y_prob, model_name: str = "") -> dict:
    return {
        "model": model_name,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def plot_roc_curves(models_preds: dict, y_true, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in models_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    return ax


def plot_pr_curves(models_preds: dict, y_true, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in models_preds.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    return ax


def plot_calibration(models_preds: dict, y_true, n_bins: int = 10, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    for name, y_prob in models_preds.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot (Reliability Diagram)")
    ax.legend()
    return ax


def assign_risk_bands(y_prob: np.ndarray) -> pd.Series:
    """
    Assign risk bands based on predicted default probability.
    Thresholds are illustrative; in practice they are set by
    the risk appetite and approval rate targets of the lender.
    """
    bands = pd.cut(
        y_prob,
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return bands


def recommend_action(risk_band: pd.Series) -> pd.Series:
    mapping = {"Low": "Approve", "Medium": "Manual Review", "High": "Decline"}
    return risk_band.map(mapping)


def risk_band_summary(y_true, y_prob) -> pd.DataFrame:
    bands = assign_risk_bands(y_prob)
    actions = recommend_action(bands)
    df = pd.DataFrame({
        "risk_band": bands,
        "recommendation": actions,
        "actual_default": y_true.values,
    })
    summary = (
        df.groupby("risk_band", observed=True)
        .agg(
            count=("actual_default", "size"),
            default_rate=("actual_default", "mean"),
            recommendation=("recommendation", "first"),
        )
        .reset_index()
    )
    summary["pct_of_applicants"] = summary["count"] / summary["count"].sum() * 100
    return summary

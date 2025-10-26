# utils/metrics.py
from __future__ import annotations

from typing import Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
)


# ---------------------------- small helpers ---------------------------- #


def round_percentage(value: float) -> float:
    """
    Convert a fraction to percentage with four decimals (e.g., 0.9123 -> 91.23).

    If input is not numeric, it is returned unchanged.
    """
    if isinstance(value, (int, float, np.floating, np.integer)):
        return round(float(value) * 100.0, 4)
    return value


def proportion_confidence_interval(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Normal-approximation CI for a proportion p in [0,1] with sample size n.

    Returns CI in the same [0,1] scale (not percentage), clipped to [0,1].
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = float(np.clip(p, 0.0, 1.0))
    z = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    half = z * np.sqrt(p * (1.0 - p) / n)
    low, high = p - half, p + half
    return float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0))


def parse_confusion_matrix(matrix_str: str) -> np.ndarray:
    """
    Parse np.array_str(...) of a confusion matrix back into a numpy array.
    Works for 2x2 or larger square matrices.
    """
    s = matrix_str.strip().replace("[[", "[").replace("]]", "]")
    rows = [r.strip().strip("[]") for r in s.split("]\n [")]
    matrix = [[int(x) for x in r.split()] for r in rows if r]
    return np.array(matrix, dtype=int)


def parse_confidence_interval(conf_interval_str: str) -> np.ndarray:
    """
    Parse np.array_str([...]) of a confidence interval back into a numpy array.
    """
    vals = conf_interval_str.strip().strip("[]").split()
    return np.array([float(v) for v in vals], dtype=float)


# ---------------------------- aggregations ---------------------------- #


def calculate_mean_results(dframe: pd.DataFrame) -> pd.Series:
    """
    Mean-aggregate a metrics DataFrame, summing confusion matrices and averaging CIs.

    Expects (optional) columns:
      - 'conf_matrix'      : stringified numpy array (via np.array_str)
      - 'conf_intervals'   : stringified numpy array (via np.array_str)
    All other numeric columns are averaged.

    Returns
    -------
    pd.Series
        A series with averaged numeric metrics and combined 'conf_matrix'/'conf_intervals' if present.
    """
    # Average numeric columns except the stringified arrays
    num = dframe.drop(columns=["conf_matrix", "conf_intervals"], errors="ignore").select_dtypes(include=[np.number])
    mean_values = num.mean()

    if "conf_matrix" in dframe.columns:
        cms = [parse_confusion_matrix(s) for s in dframe["conf_matrix"].dropna()]
        if cms:
            conf_matrix_sum = np.sum(cms, axis=0)
            mean_values["conf_matrix"] = np.array_str(conf_matrix_sum)

    if "conf_intervals" in dframe.columns:
        cis = [parse_confidence_interval(s) for s in dframe["conf_intervals"].dropna()]
        if cis:
            cis_stack = np.vstack(cis)
            mean_ci = cis_stack.mean(axis=0)
            mean_values["conf_intervals"] = np.round(mean_ci, 4)

    return mean_values


def calculate_metrics(
    results: pd.DataFrame,
    per_patient: bool = True,
    proba_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate metrics at the per-patient (or per-sample) level, then summarize.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns: 'subject', 'predicted', 'true'.
        Optionally may contain a probability column (name given by `proba_column`).
        - For binary problems, if `proba_column` is provided (probability of positive class),
          AUC will be computed from probabilities; otherwise from hard labels (not ideal).
    per_patient : bool, optional
        If True, aggregate per subject by averaging predictions and taking the first true label.
    proba_column : str or None, optional
        Column name with predicted probabilities/scores for AUC. Only used for binary.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        - first: per-patient (or per-sample) DataFrame with a 'predicted_binary' column
        - second: one-row DataFrame with summary metrics
    """
    df = results.copy()

    if per_patient:
        agg = {"predicted": "mean", "true": "first"}
        if proba_column and proba_column in df.columns:
            agg[proba_column] = "mean"
        df = df.groupby("subject", as_index=False).agg(agg)

    # binarize averaged predictions
    df["predicted_binary"] = df["predicted"].round().astype(int)

    # Compute summary
    y_true = df["true"].to_numpy()
    y_pred = df["predicted_binary"].to_numpy()
    y_score = df[proba_column].to_numpy() if (proba_column and proba_column in df.columns) else None

    summary = calculate_results(y_true, y_pred, y_score)
    return df, summary


def calculate_results(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Calculate classification metrics.

    Binary:
      - accuracy, sensitivity (recall for positive class), specificity, AUC (prefer probs)
      - also includes precision, F1, balanced_accuracy
    Multiclass:
      - accuracy, macro recall (a.k.a. sensitivity), balanced_accuracy, macro F1, macro precision
      - specificity is not well-defined; set to NaN
      - AUC requires one-vs-rest scores; computed only if `y_score` is provided as shape (N, C)

    Parameters
    ----------
    y_true : np.ndarray
    y_predicted : np.ndarray
    y_score : np.ndarray or None
        For binary: probabilities of the positive class (shape (N,)).
        For multiclass: class probabilities/scores (shape (N, C)).
    labels : Iterable[int] or None
        Explicit label order for confusion_matrix. If None, inferred.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with metrics and stringified 'conf_matrix' and 'conf_intervals'.
        Percent metrics are returned in percentage units (0..100) rounded to 4 decimals.
    """
    y_true = np.asarray(y_true).astype(int)
    y_predicted = np.asarray(y_predicted).astype(int)

    unique_labels = np.unique(y_true) if labels is None else np.array(list(labels))
    cm = confusion_matrix(y_true, y_predicted, labels=unique_labels)

    accuracy = accuracy_score(y_true, y_predicted)
    bal_acc = balanced_accuracy_score(y_true, y_predicted)

    if unique_labels.size == 2:
        # Binary metrics
        # positive class assumed to be the max label
        pos_label = unique_labels.max()
        sensitivity = recall_score(y_true, y_predicted, pos_label=pos_label)
        # specificity = TN / (TN + FP)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else np.nan

        precision = precision_score(y_true, y_predicted, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_predicted, pos_label=pos_label, zero_division=0)

        # AUC: prefer probabilities if provided
        if y_score is not None and y_score.ndim == 1:
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = np.nan
        else:
            # AUC from hard labels isn't ideal; keep for backward-compat
            try:
                auc = roc_auc_score(y_true, y_predicted)
            except Exception:
                auc = np.nan

        # CI for accuracy (normal approx)
        ci_low, ci_high = proportion_confidence_interval(accuracy, len(y_true))
        conf_intervals = np.round(np.array([ci_low, ci_high]) * 100.0, 4)

        out = pd.DataFrame(
            {
                "accuracy": [accuracy],
                "balanced_accuracy": [bal_acc],
                "sensitivity": [sensitivity],
                "specificity": [specificity],
                "precision": [precision],
                "f1": [f1],
                "auc": [auc],
                "conf_intervals": [np.array_str(conf_intervals)],
                "conf_matrix": [np.array_str(cm)],
            }
        )

        # convert fractions to percentages
        pct_cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
        out[pct_cols] = out[pct_cols].applymap(round_percentage)
        return out

    else:
        # Multiclass (macro-averaged metrics)
        sensitivity_macro = recall_score(y_true, y_predicted, average="macro", zero_division=0)
        precision_macro = precision_score(y_true, y_predicted, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_predicted, average="macro", zero_division=0)

        # Multiclass AUC (OvR) only if scores provided with proper shape
        if y_score is not None and y_score.ndim == 2 and y_score.shape[1] >= unique_labels.size:
            try:
                auc_ovr = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            except Exception:
                auc_ovr = np.nan
        else:
            auc_ovr = np.nan

        out = pd.DataFrame(
            {
                "accuracy": [accuracy],
                "balanced_accuracy": [bal_acc],
                "sensitivity_macro": [sensitivity_macro],
                "precision_macro": [precision_macro],
                "f1_macro": [f1_macro],
                "specificity": [np.nan],  # not defined cleanly for multiclass
                "auc": [auc_ovr],
                "conf_intervals": [np.array_str(np.array([np.nan, np.nan]))],
                "conf_matrix": [np.array_str(cm)],
            }
        )

        pct_cols = ["accuracy", "balanced_accuracy", "sensitivity_macro", "precision_macro", "f1_macro", "auc"]
        out[pct_cols] = out[pct_cols].applymap(round_percentage)
        return out

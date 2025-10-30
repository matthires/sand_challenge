# utils/metrics.py
from __future__ import annotations

from typing import Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import scipy.stats
import json

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


def avg_f1_score_sand(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores, class_stats = [], {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + 0.5 * (fp + fn)
        score = tp / denom if denom > 0 else 0.0
        scores.append(score)
        class_stats[int(c)] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "score": float(score)}
    return np.mean(scores), class_stats


def multiclass_specificity(y_true, y_pred, average="macro"):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        # TP: True positives for class i
        # FP: False positives for class i
        # FN: False negatives for class i
        # TN: True negatives for class i

        # For class i, treat it as the "positive" class
        # TP_i = cm[i, i]
        # FN_i = sum of row i, excluding cm[i,i]
        # FP_i = sum of col i, excluding cm[i,i]

        # TN_i = sum of all elements NOT in row i and NOT in col i

        # Calculate TN_i
        tn_i = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]

        # Calculate FP_i
        fp_i = cm[:, i].sum() - cm[i, i]

        if (tn_i + fp_i) > 0:
            specificity_i = tn_i / (tn_i + fp_i)
        else:
            specificity_i = 0.0  # Handle cases where there are no true negatives or false positives

        specificities.append(specificity_i)

    if average == "macro":
        return np.mean(specificities)
    elif average == "None":  # Return array of specificities per class
        return np.array(specificities)
    else:
        raise ValueError("Invalid averaging method. Choose 'macro' or 'None'.")


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
    num_classes: Optional[int] = None,
    labels: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate metrics at the per-patient (or per-sample) level, then summarize.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns: 'subject', 'probs', 'true'.
        Optionally may contain a probability column (name given by `proba_column`).
        - For binary problems, if `proba_column` is provided (probability of positive class),
          AUC will be computed from probabilities; otherwise from hard labels (not ideal).
    per_patient : bool, optional
        If True, aggregate per subject by averaging predictions and taking the first true label.
    proba_column : str or None, optional
        Column name with predicted probabilities/scores for AUC. Only used for binary.
    num_classes : int or None, optional (default=None)
        Specifies the total number of classes for metric computation. When set to 2, the
        function forces binary classification metrics (e.g., sensitivity, specificity),
        even if only one class is present in the data slice. If None, the number of classes
        is inferred from the data.
    labels : Iterable[int] or None, optional (default=None)
        Explicit list defining the class label space and ordering for metric calculation.
        This ensures consistent confusion matrix dimensions across folds or evaluations.
        If provided, it overrides `num_classes` for defining the label space.

    Returns
    -------
    df : pd.DataFrame
        DataFrame used for metric computation (either per-patient or per-sample),
        containing columns: 'true', 'probs', and 'predicted_label'.
    summary : pd.DataFrame
        One-row DataFrame with summary metrics (accuracy, sensitivity/recall,
        specificity for binary; macro-averaged metrics for multiclass), including
        the confusion matrix and confidence intervals.
    """
    df = results.copy()
    is_binary = (num_classes == 2) or (labels is not None and len(labels) == 2)

    if per_patient:
        if is_binary:
            # Simple scalar mean for binary (probs is a float)
            agg = {"probs": "mean", "true": "first"}
        else:
            # Elementwise mean for multiclass (probs is a vector)
            def _mean_vec(values):
                """Mean aggregation for multiclass probability vectors."""
                arrs = np.stack(values.to_numpy(), axis=0)  # shape (n_samples, num_classes)
                return np.mean(arrs, axis=0)

            agg = {"probs": _mean_vec, "true": "first"}
        df = df.groupby("subject", as_index=False).agg(agg)

    y_true = df["true"].to_numpy()

    if is_binary:
        # Binary classification -> scalar probs per row
        df["predicted_label"] = (df["probs"] >= 0.5).astype(int)
        y_pred = df["predicted_label"].to_numpy()
        y_score = df["probs"].to_numpy()

    else:
        # Multiclass -> probs is array-like per row
        probs_mat = np.stack(df["probs"].to_numpy(), axis=0)  # shape (N, C)
        df["predicted_label"] = np.argmax(probs_mat, axis=1)
        y_pred = df["predicted_label"].to_numpy()
        y_score = probs_mat  # enables multiclass AUC

    summary = calculate_results(y_true, y_pred, y_score=y_score, num_classes=num_classes, labels=labels)
    if per_patient:
        return df, summary
    return df, summary


def calculate_results(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
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

    accuracy = accuracy_score(y_true, y_predicted)
    bal_acc = balanced_accuracy_score(y_true, y_predicted)
    f1_sand, stats = avg_f1_score_sand(y_true, y_predicted)

    # If nothing provided, infer from data
    if num_classes is None and labels is None:
        uniq = np.unique(y_true.astype(int))
        if uniq.size <= 2:
            num_classes = 2
            labels = [0, 1] if set(uniq).issubset({0, 1}) else sorted(uniq.tolist())
        else:
            num_classes = int(uniq.size)
            labels = sorted(uniq.tolist())

    # If labels provided without num_classes, derive it
    if num_classes is None and labels is not None:
        num_classes = len(list(labels))

    # If num_classes provided without labels, create a default range
    if labels is None and num_classes is not None:
        labels = list(range(int(num_classes)))

    # Final guard: both must now be concrete
    if num_classes is None or labels is None:
        raise ValueError("calculate_results: could not determine class space (num_classes/labels).")

    # --- replace the old is_binary line with this safe check ---
    is_binary = (int(num_classes) == 2) or (labels is not None and len(list(labels)) == 2)

    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    if is_binary:
        # Handle 2x2 CM even if one class is absent
        if cm.shape != (2, 2):
            # pad to 2x2 if needed
            full = np.zeros((2, 2), dtype=int)
            # map existing labels into {0,1} index positions
            idx_map = {lbl: i for i, lbl in enumerate(labels)}
            if len(labels) == 1:
                # assume the single label corresponds to index 0
                full[0, 0] = cm[0, 0] if cm.size else 0
            else:
                full[: cm.shape[0], : cm.shape[1]] = cm
            cm = full

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = precision_score(y_true, y_predicted, pos_label=labels[-1], zero_division=0)
        f1 = f1_score(y_true, y_predicted, pos_label=labels[-1], zero_division=0)

        # AUC: prefer probabilities if provided
        if y_score is not None and y_score.ndim <= 1:
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_true, y_predicted)
            except Exception:
                auc = np.nan

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
                "f1_sand": f1_sand,
                "f1_sand_class_stats": [json.dumps(stats)],
            }
        )
        pct_cols = [
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "f1_sand",
            "f1_sand_class_stats",
            "auc",
        ]
        out[pct_cols] = out[pct_cols].applymap(round_percentage)
        return out

    # Multiclass branch
    sensitivity_macro = recall_score(y_true, y_predicted, average="macro", zero_division=0)
    specificity_macro = multiclass_specificity(y_true, y_predicted, average="macro")
    precision_macro = precision_score(y_true, y_predicted, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_predicted, average="macro", zero_division=0)

    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] >= len(labels):
        try:
            auc_ovr = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=labels)
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
            "specificity": [specificity_macro],
            "auc": [auc_ovr],
            "conf_matrix": [np.array_str(cm)],
            "f1_sand": f1_sand,
            "f1_sand_class_stats": [json.dumps(stats)],
        }
    )
    pct_cols = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity_macro",
        "precision_macro",
        "f1_macro",
        "f1_sand",
        "f1_sand_class_stats",
        "auc",
    ]
    out[pct_cols] = out[pct_cols].applymap(round_percentage)
    return out

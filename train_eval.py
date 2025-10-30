from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from sklearn.model_selection import StratifiedGroupKFold

from utils.config import Config
from utils.metrics import calculate_metrics, calculate_results
from utils.io import create_dir_if_not_exist, write_results_to_csv
from models.xception import Xception
from utils.callbacks import EarlyStopper  # your cleaned version
from utils.audio_processing import DataManager  # for dataframe prep (not used here, but kept for symmetry)
from utils.dataset import build_data_loader  # for dataloaders
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

# ----------------------------- Helpers ----------------------------- #


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(params, cfg: Config) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return SGD(params, lr=cfg.learning_rate, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    sch = cfg.scheduler
    if sch is None:
        return None
    if sch == "plateau":
        p = cfg.scheduler_params
        return ReduceLROnPlateau(
            optimizer,
            mode=str(p.get("mode", "min")),
            factor=float(p.get("factor", 0.5)),
            patience=int(p.get("patience", 5)),
            min_lr=float(p.get("min_lr", 1e-6)),
            verbose=False,
        )
    if sch == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=int(cfg.epochs), eta_min=float(cfg.scheduler_params.get("min_lr", 0.0))
        )
    if sch == "step":
        step_size = int(cfg.scheduler_params.get("step_size", 30))
        gamma = float(cfg.scheduler_params.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {sch}")


def build_criterion(cfg: Config) -> nn.Module:
    """
    Build the appropriate loss function based on model output and config.

    Parameters
    ----------
    cfg : Config
        Configuration object. Must include `num_classes` and optionally `loss`.

    Returns
    -------
    torch.nn.Module
        The loss function instance.
    """
    # Auto-select loss
    if cfg.num_classes <= 2:
        cfg.loss = "bce_logits"
    elif cfg.num_classes > 2:
        cfg.loss = "cross_entropy"
    else:
        # 2 logits binary (rare case)
        cfg.loss = "cross_entropy"

    # Build loss according to cfg.loss
    if cfg.loss.lower() in {"bce", "bce_logits", "binary_cross_entropy_with_logits"}:
        if cfg.class_weights is not None:
            if len(cfg.class_weights) == 1:
                pos_w = torch.tensor([cfg.class_weights[0]], dtype=torch.float32)
                return nn.BCEWithLogitsLoss(pos_weight=pos_w)
        return nn.BCEWithLogitsLoss()
    elif cfg.loss.lower() in {"ce", "cross_entropy"}:
        weight = None
        if cfg.class_weights is not None and len(cfg.class_weights) == cfg.num_classes:
            weight = torch.tensor(cfg.class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unsupported loss: {cfg.loss}")


def logits_to_preds(logits: torch.Tensor, cfg: Config) -> np.ndarray:
    """
    Convert raw model outputs to hard predictions (numpy ints).
    Binary: threshold on sigmoid at 0.5
    Multiclass: argmax
    """
    if cfg.num_classes == 1:
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        return (probs > 0.5).astype(int).reshape(-1)
    else:
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        return preds.reshape(-1)


# ----------------------------- Trainer ----------------------------- #


class NeuralNetworkModel:
    """
    Train/validate/test loop for spectrogram classification.

    Assumptions
    -----------
    - `dataset.build_data_loader` builds loaders returning dicts with keys:
      'spectrogram' (Tensor), 'label' (int), 'subject' (str)
    - Metrics helpers exist:
      - calculate_results(y_true, y_pred) -> pd.DataFrame with columns like accuracy, auc, etc.
      - calculate_metrics(df_predictions) -> (per_sample_df, per_subject_df)
      - write_results_to_csv(df, name, out_dir)
    - Config drives optimizer/scheduler/loss/early-stop settings.
    """

    def __init__(
        self, dataframe: pd.DataFrame, config: Config, test_frames: Optional[Dict[str, pd.DataFrame]] = None
    ) -> None:
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Must have 'file_path', 'label', 'subject'.
        config : Config
        test_frames : dict[str, pd.DataFrame], optional
            Mapping of dataset_label -> test dataframe. If None, skip testing.
        """
        self.cfg = config
        self.df = dataframe.reset_index(drop=True)
        self.test_frames = test_frames or {}

        # results accumulators
        self.final_results = pd.DataFrame()
        self.final_results_per_patient = pd.DataFrame()
        self.final_predictions = pd.DataFrame()
        self.final_predictions_per_patient = pd.DataFrame()

        # runtime
        self.device = resolve_device(self.cfg.device)
        set_seed(self.cfg.seed)
        torch.cuda.empty_cache()

    # -------- public API -------- #

    def train_and_validate(self, run_dir: Union[str, Path], save_models: bool = False) -> None:
        """
        K-fold train/validate (and optional test) with early stopping + scheduler.

        Parameters
        ----------
        run_dir : str or Path
            Pre-created directory for this run .
        save_models : bool, optional
            Save best checkpoint per fold as `model_fold{idx}.pt`. Default False.
        """
        create_dir_if_not_exist(Path(run_dir))
        self.cfg.validate()

        X = np.arange(len(self.df))
        y = self.df["label"].to_numpy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        groups = self.df["subject"].astype(str).to_numpy()

        self.cfg.label_encoder_classes = list(
            map(lambda x: int(x) if str(x).isdigit() else str(x), label_encoder.classes_.tolist())
        )
        self.cfg.save(run_dir)

        skf = StratifiedGroupKFold(n_splits=self.cfg.k, shuffle=True, random_state=self.cfg.seed)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y, groups), start=1):
            print(f"\nFold {fold_idx}/{self.cfg.k}")

            train_df = self.df.iloc[train_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)

            train_loader = build_data_loader(train_df, self.cfg, train=True, augmentations=self.cfg.augmentations)
            val_loader = build_data_loader(val_df, self.cfg, train=False)

            model = self._build_model().to(self.device)
            optimizer = build_optimizer(model.parameters(), self.cfg)
            scheduler = build_scheduler(optimizer, self.cfg)
            criterion = build_criterion(self.cfg).to(self.device)

            scaler = GradScaler(enabled=self.cfg.mixed_precision)
            early_stopping = EarlyStopper(patience=self.cfg.patience, min_delta=self.cfg.min_delta)

            best_val_results: Optional[pd.DataFrame] = None
            best_state_dict: Optional[Dict[str, torch.Tensor]] = None
            best_epoch = -1

            best_val_true = None
            best_val_subjects = None
            best_val_logits = None
            best_val_probs = None  # for binary
            best_val_preds_hard = None

            for epoch in range(1, self.cfg.epochs + 1):
                train_loss = self._train_one_epoch(model, train_loader, optimizer, criterion, scaler)
                val_loss, val_pred_np, val_true_np, val_subjects = self._validate_one_epoch(
                    model, val_loader, criterion
                )

                # step scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif scheduler is not None:
                    scheduler.step()

                # metrics
                val_preds_hard = logits_to_preds(torch.from_numpy(val_pred_np), self.cfg)
                # For binary single-logit (cfg.num_classes == 1), use sigmoid probs; else no score.
                val_probs_flat = (
                    torch.sigmoid(torch.from_numpy(val_pred_np)).cpu().numpy().reshape(-1)
                    if self.cfg.num_classes <= 2
                    else softmax(val_pred_np, axis=1)
                )
                if not (val_probs_flat.ndim == 2 and val_probs_flat.shape[1] == self.cfg.num_classes):
                    print("Warning: Softmax output shape mismatch for y_score.")
                    val_probs_for_auc = None

                val_df_metrics = self._compute_fold_metrics(
                    y_true=val_true_np,
                    y_pred_hard=val_preds_hard,
                    y_score=val_probs_flat,  # None for multiclass unless softmax scores are passed
                    val_loss=val_loss,
                )

                if self.cfg.num_classes == 1:
                    print(
                        f"Epoch {epoch}/{self.cfg.epochs} "
                        f"| train_loss: {train_loss:.6f} "
                        f"| val_loss: {val_loss:.6f} "
                        f"| acc: {val_df_metrics['accuracy'].item():.4f} "
                        f"| sens: {val_df_metrics['sensitivity'].item():.4f} "
                        f"| spec: {val_df_metrics['specificity'].item():.4f} "
                        f"| auc: {val_df_metrics['auc'].item():.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{self.cfg.epochs} "
                        f"| train_loss: {train_loss:.6f} "
                        f"| val_loss: {val_loss:.6f} "
                        f"| acc: {val_df_metrics['accuracy'].item():.4f} "
                        f"| bal_acc: {val_df_metrics['balanced_accuracy'].item():.4f} "
                        f"| f1_macro: {val_df_metrics['f1_macro'].item():.4f} "
                        f"| auc: {val_df_metrics['auc'].item():.4f}"
                    )

                # early stopping: store the best snapshot
                should_stop = early_stopping.early_stop(
                    {"loss": torch.tensor(val_loss, dtype=torch.float32), **val_df_metrics.to_dict("records")[0]},
                    model.state_dict(),
                )
                if best_val_results is None or val_loss < best_val_results["loss"].iloc[0]:
                    best_val_results = val_df_metrics.copy()
                    best_val_results.loc[:, "loss"] = float(val_loss)
                    best_state_dict = model.state_dict()
                    best_epoch = epoch

                    # cache arrays from the best epoch
                    best_val_true = val_true_np.copy()
                    best_val_subjects = list(val_subjects)
                    best_val_logits = val_pred_np.copy()
                    best_val_preds_hard = val_preds_hard.copy()
                    best_val_probs = (
                        torch.sigmoid(torch.from_numpy(best_val_logits)).cpu().numpy().reshape(-1)
                        if self.cfg.num_classes == 1
                        else None
                    )

                if should_stop or epoch == self.cfg.epochs:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                    # restore best
                    if best_state_dict is not None:
                        model.load_state_dict(best_state_dict)
                    break

            if best_val_true is None:
                raise RuntimeError("No best validation snapshot cached — check loaders/early stopping logic.")

            if self.cfg.num_classes == 1:
                # Binary
                preds_df = pd.DataFrame(
                    {
                        "subject": best_val_subjects,
                        "true": best_val_true.astype(int),
                        "probs": best_val_probs.astype(float),
                        "predicted_label": (best_val_probs > 0.5).astype(int),
                    }
                )

                preds_pp_df, results_per_patient = calculate_metrics(
                    preds_df,
                    per_patient=True,
                    proba_column=("probs" if "probs" in preds_df.columns else None),  # use probs for AUC
                    num_classes=2,
                    labels=[0, 1],
                )
            else:
                # Multiclass
                logits_t = torch.from_numpy(best_val_logits).float()  # [N,C]
                softmax_probs = torch.softmax(logits_t, dim=1).cpu().numpy()  # [N,C]
                pred_labels = np.argmax(softmax_probs, axis=1)

                preds_df = pd.DataFrame(
                    {
                        "subject": best_val_subjects,
                        "true": best_val_true.astype(int),
                        "probs": list(softmax_probs),
                        "predicted_label": pred_labels.astype(int),
                    }
                )

                preds_pp_df, results_per_patient = calculate_metrics(
                    preds_df,
                    per_patient=True,
                    proba_column=("probs" if "probs" in preds_df.columns else None),
                    num_classes=self.cfg.num_classes,
                    labels=list(range(self.cfg.num_classes)),
                )

            # accumulate fold results
            self.update_final_results(
                results_per_fold=best_val_results,
                results_pp_per_fold=results_per_patient,
                predictions_per_fold=preds_df,
                predictions_pp_per_fold=preds_pp_df,
                k_index=fold_idx - 1,
            )

            # save best model per fold
            if save_models and best_state_dict is not None:
                ckpt_path = run_dir / f"model_fold{fold_idx}.pt"
                torch.save(best_state_dict, ckpt_path)

            # persist running aggregates
            write_results_to_csv(self.final_results, "val_results", run_dir)
            write_results_to_csv(self.final_results_per_patient, "val_results_per_pat", run_dir)
            write_results_to_csv(self.final_predictions, "predictions", run_dir)
            write_results_to_csv(self.final_predictions_per_patient, "predictions_per_pat", run_dir)

        # Optional: evaluate on external test sets (dict of dataframes)
        if self.test_frames:
            self._test_all(run_dir)

    # -------- internals -------- #

    def _build_model(self) -> nn.Module:
        """
        Build the model according to config.

        - Binary (num_classes=1, default): return logits in forward, use BCEWithLogitsLoss.
        - Multiclass (num_classes>1): return logits, use CrossEntropyLoss.
        """
        # For training best practices, return logits from forward:
        output_activation = "none"
        model = Xception(num_classes=self.cfg.num_classes, pretrained="imagenet", output_activation=output_activation)
        return model

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: GradScaler,
    ) -> float:
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in loader:
            inputs = batch["spectrogram"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.mixed_precision):
                outputs = model(inputs)
                if self.cfg.num_classes <= 2 and self.cfg.loss == "bce_logits":
                    loss = criterion(outputs.view(-1), targets.float())
                elif self.cfg.loss == "cross_entropy":
                    loss = criterion(outputs, targets.long())
                else:
                    # fallback (shouldn't happen with valid config)
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if self.cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return running_loss / max(1, n_samples)

    def _validate_one_epoch(
        self,
        model: nn.Module,
        loader,
        criterion: nn.Module,
    ) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
        model.eval()
        running_loss = 0.0
        n_samples = 0

        all_logits: List[np.ndarray] = []
        all_targets: List[int] = []
        all_subjects: List[str] = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["spectrogram"].to(self.device, non_blocking=True)
                targets = batch["label"].to(self.device, non_blocking=True)

                outputs = model(inputs)

                if self.cfg.num_classes == 1 and self.cfg.loss == "bce_logits":
                    loss = criterion(outputs.view(-1), targets.float())
                elif self.cfg.loss == "cross_entropy":
                    loss = criterion(outputs, targets.long())
                else:
                    loss = criterion(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                n_samples += batch_size

                all_logits.append(outputs.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy().tolist())
                all_subjects.extend(batch["subject"])

        avg_loss = running_loss / max(1, n_samples)
        logits_np = np.concatenate(all_logits, axis=0)
        targets_np = np.asarray(all_targets)

        return avg_loss, logits_np, targets_np, all_subjects

    def _compute_fold_metrics(
        self, y_true: np.ndarray, y_pred_hard: np.ndarray, y_score: np.ndarray, val_loss: float
    ) -> pd.DataFrame:
        """
        Wrap your `calculate_results` and add the loss as a column.
        """
        num_classes_for_metrics = 2 if self.cfg.num_classes == 1 else self.cfg.num_classes
        labels_for_metrics = [0, 1] if num_classes_for_metrics == 2 else list(range(num_classes_for_metrics))

        res = calculate_results(
            y_true=y_true,
            y_predicted=y_pred_hard,
            y_score=y_score,  # probs for binary; None for multiclass
            num_classes=num_classes_for_metrics,
            labels=labels_for_metrics,
        )
        return res.assign(loss=float(val_loss))

    def _test_all(self, run_dir: Path) -> None:
        """
        Evaluate best model (from last fold) on each provided external test dataframe.
        TBD
        """
        pass

    # ---------------- results aggregation ---------------- #

    def update_final_results(
        self,
        results_per_fold: pd.DataFrame,
        results_pp_per_fold: pd.DataFrame,
        predictions_per_fold: pd.DataFrame,
        k_index: int,
        predictions_pp_per_fold: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Aggregate fold results and compute 'mean' rows at the end.

        Parameters
        ----------
        results_per_fold : pd.DataFrame
            Metrics per fold (per-sample level).
        results_pp_per_fold : pd.DataFrame
            Metrics per patient per fold.
        predictions_per_fold : pd.DataFrame
            Predictions (per-sample) for this fold.
        k_index : int
            Fold index (0-based).
        predictions_pp_per_fold : pd.DataFrame, optional
            Aggregated per-patient predictions for this fold (already averaged by calculate_metrics).
            Must contain columns: 'subject', 'true', 'probs' (and ideally 'predicted_label').

        """
        # --- Add fold id to predictions
        predictions_per_fold = predictions_per_fold.copy()
        predictions_per_fold["fold"] = k_index + 1

        # reorder columns if they exist
        cols = ["fold", "subject", "probs", "predicted_label", "true"]
        preds_per_patient = predictions_pp_per_fold.copy()
        predictions_per_fold = predictions_per_fold[[c for c in cols if c in predictions_per_fold.columns]]
        preds_per_patient = preds_per_patient[[c for c in cols if c in preds_per_patient.columns]]

        # ----- accumulate
        self.final_results = pd.concat([self.final_results, results_per_fold], ignore_index=True)
        self.final_results_per_patient = pd.concat(
            [self.final_results_per_patient, results_pp_per_fold], ignore_index=True
        )
        self.final_predictions = pd.concat([self.final_predictions, predictions_per_fold], ignore_index=True)

        if not hasattr(self, "final_predictions_per_patient"):
            self.final_predictions_per_patient = preds_per_patient.copy()
        else:
            self.final_predictions_per_patient = pd.concat(
                [self.final_predictions_per_patient, preds_per_patient], ignore_index=True
            )

        # ----- compute 'mean' rows at the very end
        if (k_index + 1) == self.cfg.k:
            num_classes_metrics = 2 if self.cfg.num_classes == 1 else self.cfg.num_classes
            labels_metrics = [0, 1] if num_classes_metrics == 2 else list(range(num_classes_metrics))

            # per-sample is already per-sample → per_patient=False
            _, mean_results = calculate_metrics(
                self.final_predictions,
                per_patient=False,
                num_classes=num_classes_metrics,
                labels=labels_metrics,
            )

            # per-patient predictions DF is already aggregated → per_patient=False here
            _, mean_results_per_pat = calculate_metrics(
                self.final_predictions_per_patient,
                per_patient=False,
                num_classes=num_classes_metrics,
                labels=labels_metrics,
            )

            self.final_results.loc["mean"] = mean_results.iloc[0]
            self.final_results_per_patient.loc["mean"] = mean_results_per_pat.iloc[0]

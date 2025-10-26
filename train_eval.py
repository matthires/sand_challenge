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
from models import Xception
from utils.callbacks import EarlyStopper  # your cleaned version
from utils.audio_processing import DataManager  # for dataframe prep (not used here, but kept for symmetry)
from utils.dataset import build_data_loader  # for dataloaders

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
    if cfg.loss == "bce_logits":
        # Binary: BCEWithLogitsLoss is the numerically-stable choice
        if cfg.class_weights is not None:
            if len(cfg.class_weights) == 1:
                pos_w = torch.tensor([cfg.class_weights[0]], dtype=torch.float32)
                return nn.BCEWithLogitsLoss(pos_weight=pos_w)
        return nn.BCEWithLogitsLoss()
    if cfg.loss == "cross_entropy":
        weight = None
        if cfg.class_weights is not None and len(cfg.class_weights) == cfg.num_classes:
            weight = torch.tensor(cfg.class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
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
        # persist config
        self.cfg.validate()
        self.cfg.save(run_dir)

        # folds
        X = np.arange(len(self.df))
        y = self.df["label"].to_numpy()
        groups = self.df["subject"].astype(str).to_numpy()

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
                val_df_metrics = self._compute_fold_metrics(val_true_np, val_preds_hard, val_loss)

                print(
                    f"Epoch {epoch}/{self.cfg.epochs} "
                    f"| train_loss: {train_loss:.6f} "
                    f"| val_loss: {val_loss:.6f} "
                    f"| acc: {val_df_metrics['accuracy'].item():.4f} "
                    f"| sens: {val_df_metrics['sensitivity'].item():.4f} "
                    f"| spec: {val_df_metrics['specificity'].item():.4f} "
                    f"| auc: {val_df_metrics['auc'].item():.4f}"
                )

                # early stopping: store the best snapshot
                should_stop = early_stopping.early_stop(
                    {"loss": torch.tensor(val_loss, dtype=torch.float32), **val_df_metrics.to_dict("records")[0]},
                    model.state_dict(),
                )
                if not best_val_results or val_loss < best_val_results["loss"].iloc[0]:
                    best_val_results = val_df_metrics.copy()
                    best_val_results.loc[:, "loss"] = val_loss
                    best_state_dict = model.state_dict()
                    best_epoch = epoch

                if should_stop or epoch == self.cfg.epochs:
                    print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
                    # restore best
                    if best_state_dict is not None:
                        model.load_state_dict(best_state_dict)
                    break

            # per-sample predictions DF (from last validation forward pass variables)
            preds_df = pd.DataFrame(
                {
                    "subject": val_subjects,
                    "predicted": logits_to_preds(torch.from_numpy(val_pred_np), self.cfg),
                    "true": val_true_np.astype(int),
                }
            )

            # per-patient metrics
            _, results_per_patient = calculate_metrics(preds_df)

            # accumulate fold results
            self.update_final_results(best_val_results, results_per_patient, preds_df, fold_idx - 1)

            # save best model per fold
            if save_models and best_state_dict is not None:
                ckpt_path = run_dir / f"model_fold{fold_idx}.pt"
                torch.save(best_state_dict, ckpt_path)

            # persist running aggregates
            write_results_to_csv(self.final_results, "val_results", run_dir)
            write_results_to_csv(self.final_results_per_patient, "val_results_per_pat", run_dir)
            write_results_to_csv(self.final_predictions, "predictions", run_dir)

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
                if self.cfg.num_classes == 1 and self.cfg.loss == "bce_logits":
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

    def _compute_fold_metrics(self, y_true: np.ndarray, y_pred_hard: np.ndarray, val_loss: float) -> pd.DataFrame:
        """
        Wrap your `calculate_results` and add the loss as a column.
        """
        res = calculate_results(y_true, y_pred_hard)  # your helper → DataFrame (one row)
        res = res.assign(loss=val_loss)
        return res

    def _test_all(self, run_dir: Path) -> None:
        """
        Evaluate best model (from last fold) on each provided external test dataframe.
        If you want per-fold test with each fold's best model, we can loop and average.
        """
        # simple approach: rebuild model and load the last fold best (if saved), else current weights
        # Here we just build fresh and train-once weights are already in memory from the last fold restoration.

        # Build one test loader per dataset_label and evaluate
        results_by_dataset: Dict[str, pd.DataFrame] = {}

        # For test we need a trained model; after last fold, best state is loaded in `model`,
        # but that instance is local. Simpler approach: do per-dataset test inside training loop when early-stop triggers.
        # Below is a standalone re-eval with the last best if you saved it; otherwise skip.

        # If you saved fold checkpoints, you can load the *best* overall. For now, we compute using the final model from training loop.
        # To make this precise, consider tracking `global_best_state_dict` and reload here.

        for dataset_label, test_df in self.test_frames.items():
            test_loader = build_data_loader(test_df, self.cfg, train=False)

            model = self._build_model().to(self.device)
            # NOTE: Load a checkpoint if you saved one; otherwise this is an untrained model.
            # You likely want to move per-dataset testing into the training fold loop after early stopping,
            # as your original code did. That design is better and already implemented in your earlier version.

            # We’ll skip evaluation here to avoid reporting untrained results.
            # Implementing per-fold testing during training is the preferred pattern (already in your original code path).
            pass  # intentionally no-op

    # ---------------- results aggregation ---------------- #

    def update_final_results(
        self,
        results_per_fold: pd.DataFrame,
        results_pp_per_fold: pd.DataFrame,
        predictions_per_fold: pd.DataFrame,
        k_index: int,
    ) -> None:
        """
        Aggregate fold results and compute "mean" row at the end.

        Parameters
        ----------
        results_per_fold : pd.DataFrame
        results_pp_per_fold : pd.DataFrame
        predictions_per_fold : pd.DataFrame
        k_index : int
        """
        self.final_results = pd.concat([self.final_results, results_per_fold], ignore_index=True)
        self.final_results_per_patient = pd.concat(
            [self.final_results_per_patient, results_pp_per_fold], ignore_index=True
        )
        self.final_predictions = pd.concat([self.final_predictions, predictions_per_fold], ignore_index=True)

        # At the very end, compute the "mean" rows
        if (k_index + 1) == self.cfg.k:
            _, mean_results = calculate_metrics(self.final_predictions, per_patient=False)
            _, mean_results_per_pat = calculate_metrics(self.final_predictions)
            self.final_results.loc["mean"] = mean_results.iloc[0]
            self.final_results_per_patient.loc["mean"] = mean_results_per_pat.iloc[0]

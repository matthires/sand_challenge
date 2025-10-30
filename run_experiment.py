#!/usr/bin/env python3
# run_experiment.py
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.config import Config
from utils.audio_processing import DataManager
from train_eval import NeuralNetworkModel
from utils.io import create_dir_if_not_exist


def load_config(config_path: Path | None) -> Config:
    """
    Load a Config from JSON if provided, else use defaults.
    """
    if config_path is None:
        cfg = Config()
        cfg.validate()
        return cfg
    return Config.load(config_path)


def _make_run_dir(results_root: Path, model_name: str) -> Path:
    """
    Create <cwd>/results/<model_name>/<uuid> and return it.
    """
    run_id = str(uuid.uuid4())
    run_dir = results_root / (model_name or "model") / run_id
    create_dir_if_not_exist(run_dir)
    return run_dir


def _map_labels_to_int(df: pd.DataFrame, label_order: list) -> pd.DataFrame:
    """
    Map class labels to integer ids using a fixed order.
    Works whether labels are '0'/'1' or 0/1. Unknown labels are dropped.
    """
    # normalize everything to string for mapping, then cast to int
    order_str = [str(x) for x in label_order]
    label2id = {lbl: i for i, lbl in enumerate(order_str)}

    labels_str = df["label"].astype(str)
    mask = labels_str.isin(label2id)
    out = df.loc[mask].copy()
    out.loc[:, "label"] = labels_str[mask].map(label2id).astype("int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spectrogram classification experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.json (optional; will use defaults if omitted).",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save best checkpoint per fold into the run directory.",
    )
    args = parser.parse_args()

    # 1) Load config (from JSON or defaults)
    cfg = Config.load(args.config) if args.config else Config()
    cfg.train_datasets = ["sand"]
    cfg.tasks = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU"]
    cfg.model_name = "sand"
    cfg.validate()

    # Guard: need datasets_root + train_datasets in config
    datasets_root = Path(cfg.datasets_root)
    if not datasets_root.exists():
        raise FileNotFoundError(f"Config.datasets_root not found: {datasets_root}")
    if not cfg.train_datasets:
        raise ValueError("Config.train_datasets is empty. Please set it in your config JSON.")
    if cfg.test_datasets and any(ds in cfg.train_datasets for ds in cfg.test_datasets):
        raise ValueError("A dataset cannot be both in train_datasets and test_datasets.")

    # 2) Build run directory
    results_root = Path.cwd() / "results"
    run_dir = _make_run_dir(results_root, cfg.model_name or "model")
    print(f"[run] saving artifacts to: {run_dir}")

    # 3) Discover all wavs once
    dm = DataManager(cfg)
    selected_labels = sorted(set(cfg.train_datasets + (cfg.test_datasets or [])))
    dfs = []

    for ds_label in selected_labels:
        ds_path = Path(cfg.datasets_root) / ds_label
        if not ds_path.exists():
            print(f"[warn] dataset folder not found: {ds_path} — skipping.")
            continue
        metadata_path = "sand_task_1.xlsx" if ds_label == "sand" else None
        df_info = DataManager(cfg).get_dataset_info_df(ds_path, dataset_label=ds_label, metadata_path=metadata_path)
        if df_info.empty:
            print(f"[warn] no .wav files found in dataset: {ds_label}")
            continue

        dfs.append(df_info)

    if not dfs:
        raise RuntimeError(
            f"No .wav files found for selected datasets: {selected_labels} under root: {cfg.datasets_root}"
        )

    all_info_df = pd.concat(dfs, ignore_index=True)

    # Expected columns from get_dataset_info_df: file_path, dataset_label, task, subject, label (string)
    train_df_info = all_info_df[
        all_info_df["dataset_label"].isin(cfg.train_datasets) & all_info_df["task"].isin(cfg.tasks)
    ].copy()
    if train_df_info.empty:
        raise RuntimeError(
            f"No training files after filtering. "
            f"train_datasets={cfg.train_datasets}, tasks={cfg.tasks}, root={datasets_root}"
        )

    # Establish class order from TRAIN labels (strings)
    class_order = sorted(train_df_info["label"].astype(str).unique())
    cfg.num_classes = len(class_order)

    single_logit_binary = cfg.num_classes == 2 and cfg.loss == "bce_logits"

    cfg.num_classes = 1 if single_logit_binary else cfg.num_classes
    train_df = _map_labels_to_int(train_df_info[["file_path", "subject", "label"]].copy(), class_order)

    # Keep only needed columns and map labels → ints using TRAIN order
    train_df = train_df_info[["file_path", "subject", "label"]].copy()
    train_df = _map_labels_to_int(train_df, class_order)

    # Optional TEST frames per dataset label (aligned to TRAIN classes)
    test_frames: Dict[str, pd.DataFrame] = {}
    for ds_label in cfg.test_datasets or []:
        test_df_info = all_info_df[(all_info_df["dataset_label"] == ds_label) & (all_info_df["task"].isin(cfg.tasks))][
            ["file_path", "subject", "label"]
        ].copy()
        if test_df_info.empty:
            print(f"[warn] no test files for dataset '{ds_label}' with tasks={cfg.tasks}; skipping.")
            continue
        mapped = _map_labels_to_int(test_df_info, class_order)
        if mapped.empty:
            print(f"[warn] test dataset '{ds_label}' has only labels not present in TRAIN; skipping.")
            continue
        test_frames[ds_label] = mapped

    print(
        f"[data] train rows: {len(train_df)} "
        f"| test sets: {', '.join(test_frames.keys()) if test_frames else '(none)'}"
    )

    # 4) Train/Validate (+ optional testing inside trainer)
    runner = NeuralNetworkModel(dataframe=train_df, config=cfg, test_frames=test_frames or None)
    runner.train_and_validate(run_dir=run_dir, save_models=args.save_models)

    print("[done] experiment finished.")


if __name__ == "__main__":
    main()

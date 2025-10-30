from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

from utils.config import Config


# ---------- Low-level audio helpers  ----------


def _safe_load_wav(path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """
    Load a WAV using torchaudio and return mono waveform (1, T) + sample_rate.

    Parameters
    ----------
    path : str or Path
        Audio file path.

    Returns
    -------
    (torch.Tensor, int)
        Waveform (1, T), sample rate.
    """
    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mix to mono
    return wav, sr


def _resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample waveform to target_sr if required.

    Parameters
    ----------
    wav : torch.Tensor
        (1, T) waveform.
    sr : int
        Original sample rate.
    target_sr : int
        Target sample rate.

    Returns
    -------
    torch.Tensor
        Resampled waveform (1, T').
    """
    if sr == target_sr:
        return wav
    return Resample(orig_freq=sr, new_freq=target_sr)(wav)


def _extract_subject_from_path(file_path: str) -> str:
    """
    Extract subject ID from file_path with dataset-specific quirks.

    Parameters
    ----------
    file_path : str
        Full path to the file.

    Returns
    -------
    str
        Subject identifier.
    """
    # default: parent folder name
    subject = os.path.splitext(os.path.basename(os.path.dirname(file_path)))[0]

    # dataset-specific overrides
    if "GITA" in file_path:
        # e.g., filename-based subject minus last 2 chars
        subject = os.path.splitext(os.path.basename(file_path))[0][:-2]
    if "Neurovoz" in file_path:
        # last 4 chars from filename
        subject = os.path.splitext(os.path.basename(file_path))[0][-4:]

    return subject


# ---------- DataFrame preparation ----------


class DataManager:
    """
    Utilities to prepare DataFrames and perform basic dataset bookkeeping.

    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def prepare_dataframe_from_path(self, data_paths: List[Path]) -> pd.DataFrame:
        """
        Scan class folders of WAV files and build a metadata DataFrame.

        Parameters
        ----------
        data_paths : list[Path]
            Each path corresponds to a specific class folder root.
            Class ids are assigned by the order of provided paths (0..N-1).

        Returns
        -------
        pd.DataFrame
            Columns: file_path (str), subject (str), label (int)
        """
        rows = []
        class_id = 0

        for class_root in data_paths:
            class_root = Path(class_root)
            if not class_root.exists():
                class_id += 1
                continue

            for root, _, files in os.walk(class_root):
                for fname in files:
                    if not fname.lower().endswith(".wav"):
                        continue
                    fpath = os.path.join(root, fname)
                    subject = _extract_subject_from_path(fpath)
                    rows.append(
                        {
                            "file_path": fpath,
                            "subject": subject,
                            "label": class_id,
                        }
                    )

            class_id += 1

        return pd.DataFrame(rows)

    def get_dataset_info_df(
        self, path: Union[str, Path], dataset_label: str | None = None, metadata_path: Union[str, Path] | None = None
    ) -> pd.DataFrame:
        """
        Build a DataFrame of audio files.

        Supports two modes:
        1) Multi-dataset root: pass a directory that contains dataset folders.
            (dataset_label must be None) → scans each dataset subfolder.
        2) Single dataset dir: pass a specific dataset directory and provide dataset_label.

        Expected single-dataset structure (case-insensitive for .wav):
            <dataset_dir>/<task>/<label>/<subject>/<file.wav>
        OR, if dataset_label is "sand":
            <dataset_dir>/<training>/<task>/<subject_ID>_<task>.wav

        Metadata file (e.g., metadata.xlsx) is expected to contain 'ID' and 'Class' columns.
        The 'ID' from metadata will be matched with the 'ID' extracted from audio filenames.

        Parameters
        ----------
        path : Union[str, Path]
            The base directory containing either dataset folders or a single dataset.
            If metadata_path is provided and it's relative, 'path' is used as its anchor.
        dataset_label : str | None, optional
            A label for the dataset, used for single dataset scanning. If "sand",
            the new file structure is assumed. Defaults to None.
        metadata_path : Union[str, Path] | None, optional
            Path to an .xlsx metadata file containing 'ID', 'Age', 'Sex', 'Class'.
            If None, no metadata merging occurs. Defaults to None.

        Returns
        -------
        pd.DataFrame with columns:
        - file_path (str)
        - dataset_label (str)
        - task (str)
        - subject (str)
        - label (str)
        """
        path = Path(path)

        # scanning logic for the default structure
        def _scan_single_dataset_original(ds_dir: Path, ds_label: str) -> list[dict]:
            rows: list[dict] = []
            for root, _, files in os.walk(ds_dir):
                root_path = Path(root)
                rel = root_path.relative_to(ds_dir)
                parts = list(rel.parts)

                # Need at least: task / label / subject
                if len(parts) < 3:
                    continue

                task = parts[0]
                cls_label = parts[1]
                subject = f"{cls_label}{parts[2]}"

                for f in files:
                    if f.lower().endswith(".wav"):
                        rows.append(
                            {
                                "file_path": str(root_path / f),
                                "dataset_label": ds_label,
                                "task": task,
                                "subject": subject,
                                "label": cls_label,
                            }
                        )
            return rows

        # scanning logic for the "sand" dataset_label structure
        def _scan_sand_dataset(ds_dir: Path, ds_label: str) -> list[dict]:
            rows: list[dict] = []
            for root, _, files in os.walk(ds_dir):
                root_path = Path(root)
                rel = root_path.relative_to(ds_dir)
                parts = list(rel.parts)

                if not parts:  # If it's the root directory itself, skip
                    continue

                task = parts[0]

                for f in files:
                    if f.lower().endswith(".wav"):
                        file_name_without_ext = Path(f).stem
                        subject_match = file_name_without_ext.split("_")[0]
                        subject = subject_match if subject_match.startswith("ID") else "unknown"

                        cls_label = task  # For "XY", task is also the label

                        rows.append(
                            {
                                "file_path": str(root_path / f),
                                "dataset_label": ds_label,
                                "task": task,
                                "subject": subject,
                                "label": cls_label,
                                "file_id": subject_match,  # Use subject_match as the file_id for merging
                            }
                        )
            return rows

        # Mode 1: multi-dataset root (no dataset_label provided)
        all_rows: list[dict] = []
        if dataset_label is None:
            for ds_dir in sorted(p for p in path.iterdir() if p.is_dir()):
                # For multi-dataset, we assume the original structure as there's no "XY" flag per sub-dataset
                rows = _scan_single_dataset_original(ds_dir, ds_dir.name)
                all_rows.extend(rows)
            return pd.DataFrame(all_rows)
        # Mode 2: single dataset dir
        else:
            if dataset_label == "sand":
                training_dir = path / "training"
                rows = _scan_sand_dataset(training_dir, dataset_label)
            else:
                rows = _scan_single_dataset_original(path, dataset_label)
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)

        # --- Metadata Merging Logic ---
        if metadata_path:
            # Resolve metadata_path relative to 'path' if it's not absolute
            resolved_metadata_path = Path(metadata_path)
            if not resolved_metadata_path.is_absolute():
                resolved_metadata_path = path / resolved_metadata_path

            if not resolved_metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at: {resolved_metadata_path}")
            if not resolved_metadata_path.suffix.lower() == ".xlsx":
                raise ValueError("Metadata file must be an .xlsx Excel file.")

            metadata_df = pd.read_excel(resolved_metadata_path)

            # Ensure 'ID' and 'Class' columns exist in metadata
            required_meta_cols = ["ID", "Class"]
            if not all(col in metadata_df.columns for col in required_meta_cols):
                raise ValueError(f"Metadata file must contain '{required_meta_cols}' columns.")

            # Make sure ID in metadata is string type for consistent merging
            metadata_df["ID"] = metadata_df["ID"].astype(str)
            df["file_id"] = df["file_id"].astype(str)

            # Merge the dataframes
            # Use 'file_id' from audio df and 'ID' from metadata df for merging
            df = pd.merge(df, metadata_df, left_on="file_id", right_on="ID", how="left", suffixes=("_audio", "_meta"))

            # Update the 'label' column with 'Class' from metadata
            df["label"] = df["Class"]

            # Drop temporary and redundant columns
            df = df.drop(columns=["file_id", "ID_meta", "label_audio", "Class"], errors="ignore")

        return df

    def filter_dataset(self, df: pd.DataFrame, dataset_labels: List[str], tasks: List[str]) -> pd.DataFrame:
        """
        Filter the DataFrame based on multiple dataset labels and tasks.

        Args:
            df (pd.DataFrame): DataFrame containing file paths, dataset labels, subject IDs, and labels.
            dataset_labels (List[str]): List of dataset labels to filter.
            tasks (List[str]): List of tasks to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered_df = df[(df["dataset_label"].isin(dataset_labels)) & (df["task"].isin(tasks))].copy()
        # Reset the index and drop the old index column
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df

    def preprocess_save_dataset(self, datasets_path: Path, replace_original: bool = False) -> None:
        """
        Apply VAD (Silero) to all WAV files in datasets_path.

        Parameters
        ----------
        datasets_path : Path
        replace_original : bool, optional
            If True, overwrite original files; else create a sibling `<dataset>_prep` tree.
        """
        print("Preprocessing datasets..")
        for dataset in os.listdir(datasets_path):
            dataset_path = datasets_path / dataset
            if not dataset_path.is_dir():
                continue

            output_dir = dataset_path if replace_original else dataset_path.parent / f"{dataset_path.name}_prep"
            os.makedirs(output_dir, exist_ok=True)

            for root, _, files in os.walk(dataset_path):
                relative = os.path.relpath(root, dataset_path)
                prep_dir = os.path.join(output_dir, relative)
                os.makedirs(prep_dir, exist_ok=True)

                for fname in files:
                    if not fname.lower().endswith(".wav"):
                        continue
                    in_f = Path(root) / fname
                    out_f = Path(prep_dir) / fname
                    with torch.no_grad():
                        self._remove_silence_with_silero(in_f, out_f)
        print("Processing complete.")

    def _remove_silence_with_silero(self, input_file_path: Path, output_file_path: Path) -> None:
        """
        Remove non-voiced parts using Silero VAD and save the result.

        Parameters
        ----------
        input_file_path : Path
        output_file_path : Path
        """
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False, verbose=False
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

        wav = read_audio(str(input_file_path), sampling_rate=self.config.sample_rate)
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=self.config.sample_rate,
            min_silence_duration_ms=0,
            window_size_samples=512,
            threshold=0.8,
        )

        if speech_timestamps:
            voiced = collect_chunks(speech_timestamps, wav)
        else:
            voiced = wav

        save_audio(str(output_file_path), voiced, sampling_rate=self.config.sample_rate)

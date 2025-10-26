from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

from config import Config


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

    def get_dataset_info_df(self, datasets_path: Union[str, Path]) -> pd.DataFrame:
        """
        Enumerate a directory-of-datasets into a DataFrame.

        Expected structure:
            <datasets_path>/<dataset>/<task>/<label>/<subject_name>/<file.wav>

        Parameters
        ----------
        datasets_path : str or Path

        Returns
        -------
        pd.DataFrame
            Columns: file_path, dataset_label, task, subject, label
        """
        datasets_path = Path(datasets_path)
        data = []

        for dataset_dir in sorted(datasets_path.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            for task_dir in sorted(dataset_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task = task_dir.name

                for label_dir in sorted(task_dir.iterdir()):
                    if not label_dir.is_dir():
                        continue
                    label = label_dir.name

                    for subject_dir in sorted(label_dir.iterdir()):
                        if not subject_dir.is_dir():
                            continue
                        subject = subject_dir.name

                        for f in sorted(subject_dir.iterdir()):
                            if f.suffix.lower() != ".wav":
                                continue
                            data.append(
                                {
                                    "file_path": str(f),
                                    "dataset_label": dataset,
                                    "task": task,
                                    "subject": subject,
                                    "label": label,
                                }
                            )

        return pd.DataFrame(data)

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

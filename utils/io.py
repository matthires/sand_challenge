from __future__ import annotations
from pathlib import Path
import pandas as pd


def create_dir_if_not_exist(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_results_to_csv(results: pd.DataFrame, label: str, output_path: Path) -> Path:
    """
    Write a results DataFrame to <output_path>/<label>.csv

    Parameters
    ----------
    results : pd.DataFrame
        Results to save.
    label : str
        File name stem (without .csv).
    output_path : Path
        Directory where the file will be saved.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    create_dir_if_not_exist(output_path)
    file_path = output_path / f"{label}.csv"
    results.to_csv(file_path, index=True)
    return file_path

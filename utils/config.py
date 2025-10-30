from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.augmentations import _augmentation_space


def _default_aug_space() -> Dict[str, dict]:
    # Call at instantiation time to avoid sharing mutable state across instances
    return _augmentation_space()


@dataclass
class Config:
    """
    Experiment configuration container (POJO-like).

    Parameters
    ----------
    # Training
    batch_size : int, optional
        Mini-batch size. Default is 8.
    epochs : int, optional
        Maximum number of training epochs. Default is 500.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    device : str, optional
        Torch device string (e.g., "cuda", "cpu", "cuda:0"). Default is "cuda".

    # Optimization
    learning_rate : float, optional
        Initial learning rate. Default is 5e-3.
    weight_decay : float, optional
        L2 weight decay. Default is 0.0.
    optimizer : str, optional
        Optimizer name (e.g., "adam", "adamw", "sgd"). Default is "adamw".
    scheduler : Optional[str], optional
        LR scheduler name (e.g., "cosine", "plateau", "step"). Default is "plateau".
    scheduler_params : Dict[str, Union[int, float, str]], optional
        Extra params for scheduler (e.g., T_max, factor, patience). Default sensible dict.

    # Model
    model_name : str, optional
        Model identifier (e.g., "xception"). Default is "".
    num_classes : int, optional
        Output classes. Default is 1 (binary).
    loss : str, optional
        Loss function name (e.g., "bce_logits", "cross_entropy"). Default is "bce_logits".
    class_weights : Optional[List[float]], optional
        Per-class weights for imbalanced data. Default is None.

    # Data
    sample_rate : int, optional
        Input sampling rate. Default is 16000.
    segment_length : int, optional
        Segment length in samples (or frames—define consistently). Default is 450.
    tasks : List[str], optional
        List of task IDs to include. Default is ["A"].
    note : str, optional
        Free-form note for the run. Default is "".

    # Augmentation
    augmentations : List[str], optional
        Ordered list of augmentation policy names (must exist in `aug_space` or "none"). Default is ["none"].
    random_aug_choice : bool, optional
        If True, choose one policy per sample at random; otherwise apply sequentially. Default is False.
    aug_space : Dict[str, dict], optional
        Name → params space for augmentations. Default is from `_augmentation_space()`.

    # Cross-validation
    k : int, optional
        Number of folds for K-fold CV. Default is 5.
    group_by : Optional[str], optional
        Column/key for grouped CV (e.g., patient/subject). Default is None.

    # Early stopping
    patience : int, optional
        Early stop patience in epochs. Default is 10.
    min_delta : float, optional
        Minimum meaningful improvement for early stopping. Default is 0.0.
    monitor : str, optional
        Metric to monitor (e.g., "val_loss", "val_auc"). Default is "val_loss".
    mode : str, optional
        "min" if lower is better, "max" otherwise. Default is "min".

    Notes
    -----
    - The config is JSON-serializable and can be saved/loaded for reproducibility.
    - `validate()` checks common pitfalls (e.g., invalid scheduler params, bad fields).
    """

    # Training
    batch_size: int = 8
    epochs: int = 200
    seed: int = 42
    mixed_precision: bool = True
    device: str = "cuda:0"

    # Optimization
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: Optional[str] = "plateau"
    scheduler_params: Dict[str, Union[int, float, str]] = field(
        default_factory=lambda: {"mode": "min", "factor": 0.5, "patience": 5, "min_lr": 1e-6}
    )
    grad_clip_norm: Optional[float] = None

    # Model
    model_name: str = ""
    num_classes: int = 1
    loss: str = "bce_logits"
    class_weights: Optional[List[float]] = None

    # Data
    sample_rate: int = 16000
    segment_length: int = 450
    tasks: List[str] = field(default_factory=lambda: ["A"])
    note: str = ""
    datasets_root: str = str(Path.cwd() / "data")
    train_datasets: List[str] = field(default_factory=list)
    test_datasets: List[str] = field(default_factory=list)

    # Augmentation
    augmentations: List[str] = field(default_factory=lambda: ["none"])
    random_aug_choice: bool = False
    aug_space: Dict[str, dict] = field(default_factory=_default_aug_space)

    # CV
    k: int = 5
    stratify_by: Optional[str] = None
    group_by: Optional[str] = None

    # Early stopping
    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"
    mode: str = "min"

    # ---------- Convenience methods ----------

    def to_dict(self) -> Dict:
        """
        Convert configuration to a plain dictionary.

        Returns
        -------
        dict
            JSON-serializable dictionary of config values.
        """
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        """
        Create a Config from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary of config values.

        Returns
        -------
        Config
            Instantiated configuration.
        """
        # Unknown keys will be ignored if constructor doesn't accept them; we filter them.
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)  # type: ignore[arg-type]

    def validate(self) -> None:
        """
        Validate configuration values and raise helpful errors when misconfigured.

        Raises
        ------
        ValueError
            If any configuration value is invalid.
        """
        if not self.train_datasets:
            raise ValueError("Config: train_datasets must not be empty.")
        if any(ds in self.train_datasets for ds in self.test_datasets):
            raise ValueError("Config: A dataset cannot be both in train_datasets and test_datasets.")
        if not isinstance(self.datasets_root, str) or not self.datasets_root:
            raise ValueError("datasets_root must be a non-empty string.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1.")
        if self.k < 2:
            raise ValueError("k-folds must be >= 2 for cross-validation.")
        if self.patience < 1:
            raise ValueError("patience must be >= 1.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if not isinstance(self.augmentations, list):
            raise ValueError("augmentations must be a list of strings.")
        for a in self.augmentations:
            if a != "none" and a not in self.aug_space:
                raise ValueError(f"augmentation '{a}' not found in aug_space.")
        if self.scheduler not in {None, "plateau", "cosine", "step"}:
            raise ValueError("scheduler must be one of {None, 'plateau', 'cosine', 'step'}.")
        # Minimal sanity for scheduler params
        if self.scheduler == "plateau":
            for k in ["mode", "factor", "patience"]:
                if k not in self.scheduler_params:
                    raise ValueError(f"scheduler_params must include '{k}' for 'plateau' scheduler'.")

        if self.class_weights is not None:
            if len(self.class_weights) not in {1, self.num_classes}:
                raise ValueError("class_weights must have length 1 or num_classes.")

    # ---------- Persistence ----------

    def save(self, run_dir: Union[str, Path]) -> Path:
        """
        Save configuration JSON to `<run_dir>/config.json`.

        Parameters
        ----------
        run_dir : str or Path
            Path to an **existing** experiment directory.

        Returns
        -------
        Path
            Path to the saved JSON file.

        Raises
        ------
        FileNotFoundError
            If `run_dir` does not exist.
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        cfg_path = run_dir / "config.json"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return cfg_path

    @classmethod
    def load(cls, path_or_dir: Union[str, Path]) -> "Config":
        """
        Load configuration from `<dir>/config.json` or a direct JSON file path.

        Parameters
        ----------
        path_or_dir : str or Path
            Directory containing config.json, or a JSON file path.

        Returns
        -------
        Config
            Validated configuration.
        """
        p = Path(path_or_dir)
        cfg_path = p if p.suffix.lower() == ".json" else (p / "config.json")
        with cfg_path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = cls.from_dict(d)
        cfg.validate()
        return cfg

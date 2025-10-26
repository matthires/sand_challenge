from __future__ import annotations

from typing import Dict, Union, List
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa
import librosa.display

from config import Config
from spectrograms import create_spectrogram
import augmentations as ta
from audio_processing import _safe_load_wav, _resample_if_needed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpectrogramDataset(Dataset):
    """
    Loads WAVs, optionally applies an augmentation, builds an (onset + offset) spectrogram image,
    and returns a tensor suitable for image models.

    Notes
    -----
    - Per-sample augmentation via `augmentations._apply_op`.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Config, augmentation: str | None = None) -> None:
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain columns: 'file_path', 'label', 'subject'.
        config : Config
            Experiment configuration.
        augmentation : str or None, optional
            Name of augmentation to apply (must exist in config.aug_space or be "none").
        """
        self.df = dataframe.reset_index(drop=True)
        self.cfg = config
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.df)

    def _load_wav_segment(self, idx: int) -> Tensor:
        """
        Load WAV -> resample -> take onset + offset, concatenated.
        Returns mono waveform (1, T_total).
        """
        fpath = self.df.at[idx, "file_path"]
        wav, sr = _safe_load_wav(fpath)
        wav = _resample_if_needed(wav, sr, self.cfg.sample_rate)

        seg_len = int((self.cfg.segment_length / 1000.0) * self.cfg.sample_rate)
        if seg_len <= 0:
            raise ValueError("segment_length must be > 0 ms.")

        # pad if too short
        if wav.size(1) < seg_len:
            wav = torch.nn.functional.pad(wav, (0, seg_len - wav.size(1)))

        onset = wav[:, :seg_len]
        offset = wav[:, -seg_len:] if wav.size(1) >= seg_len else wav[:, :seg_len]
        return torch.cat([onset, offset], dim=1)  # (1, 2*seg_len)

    def _apply_augmentation(self, wav: Tensor) -> Tensor:
        """
        Apply per-sample augmentation if requested.
        """
        if not self.augmentation or self.augmentation == "none":
            return wav
        aug_wav, _ = ta._apply_op(
            audio_data=wav,
            sample_rate=self.cfg.sample_rate,
            op_name=self.augmentation,
            aug_space=self.cfg.aug_space,
        )
        return aug_wav

    @staticmethod
    def _spectrogram_to_pil_image(spec: Tensor, sample_rate: int) -> Image.Image:
        """
        Render a spectrogram (freq x time) into a PIL RGB image via matplotlib.
        """
        fig, ax = plt.subplots()
        plt.axis("off")
        fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
        ax.margins(0, 0)
        fig.tight_layout(pad=0)

        librosa.display.specshow(spec.detach().cpu().numpy(), sr=sample_rate, y_axis="log")

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(int(h), int(w), 3)
        plt.close(fig)

        return Image.fromarray(image, mode="RGB")

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int, str]]:
        wav = self._load_wav_segment(idx)
        wav = self._apply_augmentation(wav)

        spec = create_spectrogram(
            samples=wav, sample_rate=self.cfg.sample_rate, stride_ms=5.0, window_ms=60.0, eps=1e-11
        )

        img = self._spectrogram_to_pil_image(spec, self.cfg.sample_rate)

        preprocess = Compose(
            [
                Resize((299, 299)),  # keep the model’s expected input size
                ToTensor(),
            ]
        )

        spectrogram_tensor = preprocess(img).to(DEVICE)

        return {
            "spectrogram": spectrogram_tensor,
            "label": int(self.df.at[idx, "label"]),
            "subject": str(self.df.at[idx, "subject"]),
        }


def build_data_loader(
    dataframe: pd.DataFrame,
    config: Config,
    train: bool = True,
    augmentations: List[str] = [None],
) -> DataLoader:
    """
    Build a DataLoader. For training, concatenate copies with different augmentations.

    Parameters
    ----------
    dataframe : pd.DataFrame
    config : Config
    train : bool, optional
        If True, shuffle and allow multiple augmentation variants via ConcatDataset.
    augmentations : list[str], optional
        List of augmentation op names (must be keys in config.aug_space or "none").

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if train:
        datasets = [SpectrogramDataset(dataframe, config, augmentation=aug) for aug in augmentations]
        dataset = ConcatDataset(datasets)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    else:
        dataset = SpectrogramDataset(dataframe, config, augmentation=None)
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

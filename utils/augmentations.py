from __future__ import annotations

from typing import Dict, Tuple, Any
import numpy as np
import torch
from torch import Tensor

# Hybrid stack (kept minimal & stable)
import augly.audio as audaugs
import colorednoise as cn
from torch_audiomentations import PitchShift as TA_PitchShift
from audiomentations import HighPassFilter, LowPassFilter
from torchaudio.transforms import Resample, Vol

from scipy.signal import butter, lfilter


# ----------------------------- Public API ----------------------------- #


def get_aug_space() -> Dict[str, Dict[str, Any]]:
    """
    Define the augmentation parameter search space.

    Returns
    -------
    dict
        Mapping: op_name -> {param_name: iterable of candidate values}.

    Notes
    -----
    - Values are lightweight iterables (numpy arrays), later condensed to a single
      concrete value by `_augmentation_space(apply_random=...)`.
    """
    return {
        # "band_pass": {"low_high": ((500, 3000),)},  # enable & implement param if you want
        "colored_noise": {"power_spectrum": np.arange(3.0, 5.0, 0.25)},
        "harmonic": {"margin_size": np.arange(1.0, 3.0, 0.5)},
        "high_pass": {"cutoff_hz": np.arange(3000, 5000, 500)},
        "low_pass": {"cutoff_hz": np.arange(500, 1500, 250)},
        "noise": {"snr_level": np.arange(10, 30, 5)},
        "normalize": {"norm_type": _get_norm_values()},
        "percussive": {"margin_size": np.arange(1.0, 3.0, 0.5)},
        "pitch_shift": {"num_steps": np.arange(5.0, 15.0, 2.5)},
        "resample": {"factor": np.arange(0.8, 1.3, 0.1)},
        "slow_down": {"factor": np.arange(0.5, 0.9, 0.1)},
        "speed_up": {"factor": np.arange(1.1, 2.0, 0.1)},
        "time_shift": {"num_steps": np.arange(10, 100, 10)},
        "volume_down": {"factor": np.arange(0.5, 0.9, 0.1)},
        "volume_up": {"factor": np.arange(1.1, 2.0, 0.1)},
        "none": {"none": np.zeros(1)},
    }


def _augmentation_space(apply_random: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Collapse the augmentation space to a single value per parameter.

    Parameters
    ----------
    apply_random : bool, optional
        If True, sample randomly from each parameter's iterable.
        If False, take the first value. Default is False.

    Returns
    -------
    dict
        Mapping: op_name -> {param_name: value}
    """
    space = get_aug_space()
    rng = np.random.default_rng()

    chosen: Dict[str, Dict[str, float]] = {}
    for op, pspace in space.items():
        chosen[op] = {}
        for pname, values in pspace.items():
            vals = list(values)
            v = rng.choice(vals) if apply_random else vals[0]
            chosen[op][pname] = float(v)
    return chosen


def _apply_op(audio_data: Tensor, sample_rate: int, op_name: str, aug_space: Dict) -> Tuple[Tensor, int]:
    """
    Apply a single augmentation by name (per-sample).

    Parameters
    ----------
    audio_data : torch.Tensor
        Waveform with shape (channels, samples). If (samples,), it is treated as mono.
    sample_rate : int
        Sampling rate in Hz.
    op_name : str
        Name of the augmentation to apply (key from `get_aug_space()`).
    aug_space : dict
        Concrete parameter dict for `op_name`, e.g., {"num_steps": 4.0}.

    Returns
    -------
    (torch.Tensor, int)
        Augmented waveform (channels, samples) and (possibly updated) sample rate.

    Notes
    -----
    - Some libraries require mono numpy input; in those cases we mix down to mono,
      run the op, and return mono as (1, samples).
    """
    params = aug_space.get(op_name, {})
    aug_value = next(iter(params.values()), None)
    return _apply_op_core(audio_data, sample_rate, op_name, aug_value)


def bandpass_filter(data: np.ndarray, rate: int, lowcut: float, highcut: float, order: int = 2) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Waveform as (samples,) or (channels, samples).
    rate : int
        Sampling rate (Hz).
    lowcut : float
        Low cutoff (Hz).
    highcut : float
        High cutoff (Hz).
    order : int, optional
        Filter order. Default is 2.

    Returns
    -------
    np.ndarray
        Filtered waveform with same shape as input.
    """
    nyq = 0.5 * rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    if data.ndim == 1:
        return lfilter(b, a, data)
    if data.ndim == 2:
        return np.vstack([lfilter(b, a, ch) for ch in data])
    raise ValueError("bandpass_filter expects 1D or 2D array (channels, samples).")


# ----------------------------- Internals ----------------------------- #


def _get_norm_values() -> list[float]:
    """Common norm values for AugLy normalize."""
    return [float(np.inf), float(-np.inf), 0.0, 1.0, 2.0]


def _ensure_2d(x: Tensor) -> Tensor:
    """Ensure (channels, samples)."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError("audio_data must have shape (samples,) or (channels, samples).")


def _to_numpy_mono(x: Tensor) -> np.ndarray:
    """Mix to mono if needed and return float32 numpy 1D."""
    x2d = _ensure_2d(x).detach().cpu().numpy()
    mono = x2d.mean(axis=0) if x2d.shape[0] > 1 else x2d[0]
    return mono.astype(np.float32, copy=False)


def _apply_op_core(x_in: Tensor, sr_in: int, op: str, val: float | None) -> Tuple[Tensor, int]:
    x = _ensure_2d(x_in).to(torch.float32)
    sr = int(sr_in)

    if op == "band_pass":
        data_np = bandpass_filter(_to_numpy_mono(x), rate=sr, lowcut=500.0, highcut=3000.0)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "colored_noise":
        c, t = x.shape
        # Generate colored noise of length t and broadcast per-channel
        noise = torch.from_numpy(cn.powerlaw_psd_gaussian(float(val), t).astype(np.float32))
        if c > 1:
            noise = noise.expand(c, -1)
        y = x + noise

    elif op == "harmonic":
        data_np, _ = audaugs.harmonic(_to_numpy_mono(x), kernel_size=31, power=2.0, margin=float(val))
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "high_pass":
        # audiomentations expects numpy mono; we set a safe range with p=1.0
        hp = HighPassFilter(min_cutoff_freq=20.0, max_cutoff_freq=float(val), p=1.0)
        data_np = hp(_to_numpy_mono(x), sample_rate=sr)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "low_pass":
        lp = LowPassFilter(min_cutoff_freq=float(val), max_cutoff_freq=max(100.0, sr / 2 - 100.0), p=1.0)
        data_np = lp(_to_numpy_mono(x), sample_rate=sr)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "noise":
        data_np, _ = audaugs.add_background_noise(_to_numpy_mono(x), snr_level_db=float(val))
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "normalize":
        data_np, _ = audaugs.normalize(_to_numpy_mono(x), norm=float(val), axis=0)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "percussive":
        data_np, _ = audaugs.percussive(_to_numpy_mono(x), kernel_size=31, power=2.0, margin=float(val))
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "pitch_shift":
        # torch_audiomentations expects (B, C, T)
        n_steps = float(val)
        ta_ps = TA_PitchShift(sample_rate=sr, n_steps=n_steps, p=1.0)
        y = ta_ps(x.unsqueeze(0)).squeeze(0)

    elif op == "resample":
        factor = float(val)
        new_sr = max(1, int(round(sr * factor)))
        resample = Resample(orig_freq=sr, new_freq=new_sr)
        y = resample(x)
        sr = new_sr

    elif op == "slow_down":
        factor = float(val)  # < 1.0
        data_np, _ = audaugs.speed(_to_numpy_mono(x), factor=factor)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "speed_up":
        factor = float(val)  # > 1.0
        data_np, _ = audaugs.speed(_to_numpy_mono(x), factor=factor)
        y = torch.from_numpy(data_np).unsqueeze(0)

    elif op == "time_shift":
        shift = int(float(val))
        y = torch.from_numpy(np.roll(x.detach().cpu().numpy(), shift=shift, axis=-1)).to(x.dtype)

    elif op == "volume_up":
        vol = Vol(gain=float(val), gain_type="amplitude")
        y = vol(x)

    elif op == "volume_down":
        vol = Vol(gain=float(val), gain_type="amplitude")
        y = vol(x)

    elif op == "none":
        y = x

    else:
        raise ValueError(f"Unrecognized augmentation op: '{op}'.")

    return y.to(torch.float32), sr

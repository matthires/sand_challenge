from __future__ import annotations
import torch
from torch import Tensor


def create_spectrogram(
    samples: Tensor,
    sample_rate: int = 16000,
    stride_ms: float = 5.0,
    window_ms: float = 60.0,
    eps: float = 1e-11,
) -> Tensor:
    """
    Create a log-power spectrogram using framing + rFFT.

    Parameters
    ----------
    samples : torch.Tensor
        Waveform tensor of shape (channels, samples) or (samples,). Mono vectors are
        treated as (1, samples).
    sample_rate : int, optional
        Sampling rate in Hz. Default is 16000.
    stride_ms : float, optional
        Stride duration in milliseconds. Default is 5.0.
    window_ms : float, optional
        Window duration in milliseconds. Default is 60.0.
    eps : float, optional
        Small constant to avoid log(0). Default is 1e-11.

    Returns
    -------
    torch.Tensor
        Log-power spectrogram of shape (freq_bins, frames).

    Notes
    -----
    - Multi-channel inputs are averaged after the power spectrum.
    """
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)  # (1, T)

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Trim so that unfold works cleanly
    truncate_size = (samples.size(1) - window_size) % stride_size
    if truncate_size > 0:
        samples = samples[:, :-truncate_size]

    num_windows = (samples.size(1) - window_size) // stride_size + 1
    if num_windows <= 0:
        raise ValueError("Window/stride combination yields zero frames. Adjust window_ms/stride_ms.")

    # (C, T) -> (C, frames, window_size)
    windows = samples.unfold(1, window_size, stride_size)

    # Apply Hann window per-frame
    weighting = torch.hann_window(window_size, dtype=samples.dtype, device=samples.device)
    windows = windows * weighting  # broadcast over last dim

    # rFFT over last dim -> power spectrum
    fft = torch.fft.rfft(windows, dim=-1)
    power_spectrum = torch.abs(fft) ** 2  # (C, frames, freq_bins)

    # Scale
    scale = torch.sum(weighting**2) * sample_rate
    power_spectrum[..., 1:-1] *= 2.0 / scale
    power_spectrum[..., (0, -1)] /= scale

    # Merge channels by averaging
    power_spectrum = power_spectrum.mean(dim=0)  # (frames, freq_bins)

    log_spectrogram = torch.log(power_spectrum + eps)  # (frames, freq_bins)
    return log_spectrogram.T  # (freq_bins, frames)

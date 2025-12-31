"""
Fractional delay and sub-sample shifting utilities.
"""
import numpy as np

def shift_signal_sinc(x: np.ndarray, delay: float, N: int = 21) -> np.ndarray:
    """
    Apply a fractional delay to a signal using a windowed sinc filter.

    Args:
        x: Input signal (1D array).
        delay: Fractional delay in samples (positive = shift right / delay).
        N: Filter length (must be odd).

    Returns:
        Shifted signal.
    """
    if delay == 0:
        return x.copy()

    n = np.arange(N) - (N - 1) / 2
    h = np.sinc(n - delay) * np.hamming(N)
    h /= np.sum(h) # Normalize

    # Convolve
    # mode='same' keeps the output size the same, but we need to handle edge effects.
    # The delay is baked into the kernel 'h'.
    return np.convolve(x, h, mode='same')

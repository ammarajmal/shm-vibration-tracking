#!/usr/bin/env python3
"""
Compare Frequency Domain (FFT) of Baseline vs Robust pipelines.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from shmtrack.io.schema import T, Y

def compute_psd(t, y, fs=60.0):
    """
    Compute PSD using Welch's method.
    Handle NaNs by linear interpolation.
    """
    # Interpolate NaNs
    if np.any(np.isnan(y)):
        # print("Interpolating NaNs for FFT...")
        valid = ~np.isnan(y)
        if not np.any(valid):
            return np.array([]), np.array([])
        y = np.interp(np.arange(len(y)), np.arange(len(y))[valid], y[valid])

    f, Pxx = welch(y, fs=fs, nperseg=256)
    return f, Pxx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline CSV")
    parser.add_argument("--robust", type=Path, required=True, help="Robust/Fused CSV")
    parser.add_argument("--out", type=Path, default="fft_compare.png")
    parser.add_argument("--fs", type=float, default=60.0)

    args = parser.parse_args()

    df_base = pd.read_csv(args.baseline)
    df_rob = pd.read_csv(args.robust)

    # Extract signal
    # Assume T is roughly uniform or we just take values
    y_base = df_base[Y].values
    y_rob = df_rob[Y].values # Fused might not have gaps, but let's check

    # Compute PSD
    f_b, p_b = compute_psd(None, y_base, fs=args.fs)
    f_r, p_r = compute_psd(None, y_rob, fs=args.fs)

    # Plot
    plt.figure(figsize=(10, 6))

    if len(f_b) > 0:
        plt.semilogy(f_b, p_b, 'k-', alpha=0.5, label="Baseline (Tag Only)")

    if len(f_r) > 0:
        plt.semilogy(f_r, p_r, 'b-', linewidth=2, label="Robust (Fused)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m^2/Hz)")
    plt.title("Frequency Domain Comparison: Signal Fidelity")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)

    plt.savefig(args.out)
    print(f"Saved FFT comparison to {args.out}")

if __name__ == "__main__":
    main()

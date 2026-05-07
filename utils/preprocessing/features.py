import os
import sys
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed

def extract_band_power_vect(window, sfreq=256):
    """Vectorized PSD for standard bands across all channels."""
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}
    # PSD for all channels at once
    freqs, psd = welch(window, sfreq, axis=1, nperseg=sfreq)
    features = []
    for f_min, f_max in bands.values():
        idx = np.logical_and(freqs >= f_min, freqs <= f_max)
        # avg power in band for each channel
        features.append(np.mean(psd[:, idx], axis=1))
    return np.hstack(features) # Order: [ch1_delta, ch2_delta... ch1_theta...]

def extract_temporal_vect(window):
    """Fast statistical features using NumPy vectorization."""
    return np.hstack([
        np.mean(window, axis=1), 
        np.std(window, axis=1), 
        skew(window, axis=1), 
        kurtosis(window, axis=1)
    ])

def extract_petrosian_fd(window):
    """Fast fractal dimension (alternative to Hurst) based on zero-crossings."""
    diff = np.diff(window, axis=1)
    # count sign changes
    n_zeros = np.sum(diff[:, 1:] * diff[:, :-1] < 0, axis=1)
    n = window.shape[1]
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_zeros)))

def compute_window_features(w, sfreq):
    """Worker function to process a single window's features."""
    f_list = [
        extract_band_power_vect(w, sfreq),
        extract_temporal_vect(w),
        np.sum(np.abs(np.diff(w, axis=1)), axis=1), # Line length
        np.corrcoef(w)[np.triu_indices(w.shape[0], k=1)], # Pearson corr
        extract_petrosian_fd(w) # Fast Nonlinear
    ]
    return np.concatenate(f_list)

def get_feature_names(channels):
    """Generates column names matching the order of the combined vector."""
    names = []
    # Bands
    for ch in channels:
        for b in ['delta', 'theta', 'alpha', 'beta']: names.append(f"{ch}_{b}_pwr")
    # Stats
    for stat in ['mean', 'std', 'skew', 'kurt']:
        for ch in channels: names.append(f"{ch}_{stat}")
    # Line Length
    for ch in channels: names.append(f"{ch}_line_length")
    # Connectivity (Pearson)
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i+1:]: names.append(f"corr_{ch1}_{ch2}")
    # Nonlinear
    for ch in channels: names.append(f"{ch}_entropy")
    for ch in channels: names.append(f"{ch}_hurst")
    return np.array(names)
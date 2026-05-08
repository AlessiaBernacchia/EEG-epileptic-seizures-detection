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

def extract_ratios(band_powers):
    """ TODO"""
    # band_powers is [delta, theta, alpha, beta]
    d, t, a, b = band_powers
    
    # Avoid division by zero with a small epsilon
    eps = 1e-6
    
    slow_to_fast = (d + t) / (a + b + eps)
    theta_alpha = t / (a + eps)
    delta_alpha = d / (a + eps)
    beta_alpha = b / (a + eps)
    
    return np.hstack([slow_to_fast, theta_alpha, delta_alpha, beta_alpha])

def extract_band_power_vect_and_ratios(window, sfreq=256):
    """Vectorized PSD for standard bands across all channels and calculate also the relevant ratios for each."""
    band_powers = extract_band_power_vect(window, sfreq)
    extract_ratios = extract_ratios(band_powers)
    return np.hstack([band_powers, extract_ratios])

def extract_band_power_and_ratios(window, sfreq=256):
    """Vectorized PSD bands and clinical ratios for all channels."""
    # window shape: (channels, samples)
    freqs, psd = welch(window, sfreq, axis=1, nperseg=sfreq)
    
    bands = {'d': (0.5, 4), 't': (4, 8), 'a': (8, 12), 'b': (12, 30)}
    powers = {}
    for b, (f_min, f_max) in bands.items():
        idx = np.logical_and(freqs >= f_min, freqs <= f_max)
        powers[b] = np.mean(psd[:, idx], axis=1) # (n_channels,)
    
    eps = 1e-6
    # Calculate Ratios
    stf = (powers['d'] + powers['t']) / (powers['a'] + powers['b'] + eps)
    tar = powers['t'] / (powers['a'] + eps)
    dar = powers['d'] / (powers['a'] + eps)
    bar = powers['b'] / (powers['a'] + eps)
    
    # Concatenate: [delta_all_chs, theta_all_chs..., stf_all_chs...]
    return np.hstack([powers['d'], powers['t'], powers['a'], powers['b'], stf, tar, dar, bar])

def extract_hjorth_params(window):
    """Activity, Mobility, and Complexity for all channels."""
    # Activity: Variance
    activity = np.var(window, axis=1)
    
    # Derivatives
    d1 = np.diff(window, axis=1)
    d2 = np.diff(d1, axis=1)
    
    # Mobility
    var_d1 = np.var(d1, axis=1)
    mobility = np.sqrt(var_d1 / (activity + 1e-6))
    
    # Complexity
    var_d2 = np.var(d2, axis=1)
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-6)) / (mobility + 1e-6)
    
    return np.hstack([activity, mobility, complexity])

def extract_zcr(window):
    """Zero Crossing Rate per channel."""
    # Find signs, diff them, and count non-zeros where sign changed
    return np.mean(np.diff(np.sign(window), axis=1) != 0, axis=1)

def compute_window_features(w, sfreq):
    """Unified worker function for all window-level features."""
    f_list = [
        extract_band_power_and_ratios(w, sfreq), # Bands + Ratios
        extract_temporal_vect(w),                # Stats
        extract_hjorth_params(w),                # Hjorth
        extract_zcr(w),                         # ZCR
        np.sum(np.abs(np.diff(w, axis=1)), axis=1), # Line length
        np.corrcoef(w)[np.triu_indices(w.shape[0], k=1)], # Pearson
        extract_petrosian_fd(w)                  # Fractal Dim
    ]
    return np.concatenate(f_list)

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
    """
    Unified worker function for all window-level features.
    The order here MUST match the order in get_feature_names.
    """
    # 1. Bands and Ratios: [4 bands * chs] + [4 ratios * chs]
    # This uses the optimized vectorized function
    f_bands_ratios = extract_band_power_and_ratios(w, sfreq)
    
    # 2. Stats: [4 stats * chs]
    f_stats = extract_temporal_vect(w)
    
    # 3. Hjorth: [3 params * chs]
    f_hjorth = extract_hjorth_params(w)
    
    # 4. ZCR: [1 val * chs]
    f_zcr = extract_zcr(w)
    
    # 5. Line Length: [1 val * chs]
    f_ll = np.sum(np.abs(np.diff(w, axis=1)), axis=1)
    
    # 6. Pearson Correlation: [N*(N-1)/2 values]
    # Correlation between channels (Connectivity)
    f_corr = np.corrcoef(w)[np.triu_indices(w.shape[0], k=1)]
    
    # 7. Fractal Dimension: [1 val * chs]
    f_fractal = extract_petrosian_fd(w)

    # Combine all into a single flat vector
    return np.concatenate([
        f_bands_ratios, 
        f_stats, 
        f_hjorth, 
        f_zcr, 
        f_ll, 
        f_corr, 
        f_fractal
    ])


def get_feature_names(channels):
    """Generates names matching the order in compute_window_features."""
    names = []

    # 1. Bands (4 bands * channels)
    for b in ['delta', 'theta', 'alpha', 'beta']:
        for ch in channels: names.append(f"{ch}_{b}_pwr")
        
    # 2. Ratios (4 ratios * channels)
    for r in ['stf', 'tar', 'dar', 'bar']:
        for ch in channels: names.append(f"{ch}_{r}_ratio")
        
    # 3. Stats (4 stats * channels)
    # Order in extract_temporal_vect is: mean, std, skew, kurt
    for s in ['mean', 'std', 'skew', 'kurt']:
        for ch in channels: names.append(f"{ch}_{s}")
        
    # 4. Hjorth (3 params * channels)
    for h in ['hjorth_act', 'hjorth_mob', 'hjorth_comp']:
        for ch in channels: names.append(f"{ch}_{h}")
        
    # 5. ZCR (1 per channel)
    for ch in channels: names.append(f"{ch}_zcr")
    
    # 6. Line Length (1 per channel)
    for ch in channels: names.append(f"{ch}_line_length")
    
    # 7. Pearson (N*(N-1)/2 values total)
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i+1:]: 
            names.append(f"corr_{ch1}_{ch2}")
            
    # 8. Fractal (1 per channel)
    for ch in channels: names.append(f"{ch}_petrosian_fd")
    
    return np.array(names)
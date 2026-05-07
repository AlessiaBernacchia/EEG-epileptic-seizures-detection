import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_window(index_window, windows, channels, times, offset=5, sfreq=256, patient_name='chb04', legend=False):
    """
    Plots a specific window from the 3D tensor.
    The X-axis is converted from samples to real-time seconds.
    """
    plt.figure(figsize=(15, 10))
    
    # select the specific window
    window_to_plot = windows[index_window]
    start_time = times[index_window]
    
    # calculate time vector for the X-axis (Window Start + (sample_index / SFREQ))
    num_samples = window_to_plot.shape[1]
    time_axis = np.linspace(start_time, start_time + (num_samples / sfreq), num_samples)

    # plot each channel with a vertical offset to prevent overlapping
    for i, ch_name in enumerate(channels):
        plt.plot(time_axis, window_to_plot[i] + (i * offset), label=ch_name, linewidth=0.8)

    plt.title(f"EEG Visualization: Window {index_window} (Starts at {start_time:.2f}s) - Patient {patient_name}")
    plt.xlabel("Time (Seconds from start of recording)")
    plt.ylabel("EEG Channels (scaled with vertical offset)")
    
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize='small', ncol=2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
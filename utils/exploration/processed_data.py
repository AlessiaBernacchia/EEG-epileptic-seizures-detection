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
    calculated_offset = np.max(window_to_plot) - np.min(window_to_plot) # safe offset to avoid overlapping
    start_time = times[index_window]
    
    # calculate time vector for the X-axis (Window Start + (sample_index / SFREQ))
    num_samples = window_to_plot.shape[1]
    time_axis = np.linspace(start_time, start_time + (num_samples / sfreq), num_samples)

    # plot each channel with a vertical offset to prevent overlapping
    for i, ch_name in enumerate(channels):
        plt.plot(time_axis, window_to_plot[i] - (i * calculated_offset), label=ch_name, linewidth=0.8)

    plt.title(f"EEG Visualization: Window {index_window} (Starts at {start_time:.2f}s) - Patient {patient_name}")
    plt.xlabel("Time (Seconds from start of recording)")
    plt.ylabel("EEG Channels (scaled with vertical offset)")
    
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0), fontsize='small', ncol=2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_single_file_info(file, keys=[]):
    """ Print general info about one single file """
    with np.load(file) as data:
        keys_available = data.files
        print(f'Available keys for the file are: {keys_available}')
        if len(keys) == 0:
            for k in keys_available:
                print(f'Shape of {k}: {data[k].shape}')
        elif len(keys) == 2 and keys[0] in ['X_train', 'X_val', 'X_test']:
            X = data[keys[0]]
            y = data[keys[1]]
            feat_names = data['feature_names']
            print(f'The shape of the set is: {X.shape}')
            print(f'The shape of the target label is: {y.shape}')
            print(f'The features that we have are {feat_names.shape}')
        else:
            for k in keys:
                print(f'Shape of {k}: {data[k].shape}')
        
    print('\n\n')

def get_agglomerated_labels(file_list, label_key='y_val'):
    """Extracts and concatenates labels from a list of npz files."""
    labels_list = []
    for f in file_list:
        with np.load(f) as data:
            labels_list.append(data[label_key])
    return np.concatenate(labels_list, axis=0)

def plot_class_distribution(labels, title="Class Distribution", save=False, save_path=None, name='class_distribution.png'):
    """Prints stats and plots a bar chart for the given label array."""
    classes, counts = np.unique(labels, return_counts=True)
    
    # Print numerical stats
    print(f"--- {title} ---")
    for c, k in zip(classes, counts):
        print(f'Class {c}: {k} samples ({(k/len(labels))*100:.2f}%)')
    
    # Plotting
    fig = plt.figure(figsize=(6, 4))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'] # Added more colors just in case
    bars = plt.bar(classes.astype(str), counts, color=colors[:len(classes)])
    
    plt.title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', 
                 ha='center', va='bottom', fontweight='bold')
    
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path / name)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
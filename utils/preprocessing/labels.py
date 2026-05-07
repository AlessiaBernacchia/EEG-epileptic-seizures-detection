import numpy as np
from joblib import Parallel, delayed

def compute_window_labels(window_time, seizure_intervals, window_sec=10):
    """
    Computes labels for both tasks based on the window start time.
    - TTS: Time-To-Seizure
    - 0: seizure
    - -1: no more seizures
    """
    # Task 01: Binary classification (1 if window overlaps with any seizure)
    is_seizure = 0
    window_end = window_time + window_sec
    
    for start, end in seizure_intervals:
        # check for any overlap between window and seizure interval
        if not (window_end <= start or window_time >= end):
            is_seizure = 1
            break
            
    # Task 02: Forecasting (Time-to-Seizure in seconds)
    # If currently in seizure, TTS is 0. Otherwise, time until next seizure starts.
    future_seizures = [start for start, end in seizure_intervals if start > window_time]
    
    if is_seizure:
        tts = 0.0
    elif not future_seizures:
        tts = -1.0 # placeholder for "no more seizures in this file"
    else:
        tts = min(future_seizures) - window_time
        
    return np.array([is_seizure, tts])
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def import_npz(patient_id: int, num_file: int, folder_path):
    """ Import a specific num_file file from a specific patient given the patient_id"""
    # define path file
    p_name = f"chb{str(patient_id).zfill(2)}"
    n_file = str(num_file).zfill(2)
    path_file = folder_path / p_name / f"{p_name}_{n_file}.npz"
    # load data
    data = np.load(path_file)
    return data
import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib

cur_dir = Path(os.getcwd()).resolve()
proj_dir = next(parent for parent in [cur_dir, *cur_dir.parents] if (parent / "utils").exists() and (parent / "data").exists())

SPLIT_PATH = proj_dir / "data" / "split" / "task_1" 
CHANNEL_INDEX = 3


def selected_feature_indices(feature_names, channel_name):
    return [i for i, name in enumerate(feature_names) if str(name).startswith(f"{channel_name}_")]

def make_frame(X, y, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df["is_seizure"] = np.asarray(y).astype(int)
    return df

def load_file(file_name, channel_index=CHANNEL_INDEX, path=SPLIT_PATH):
    patient = file_name.split("_")[0]
    data = np.load(path / f"{file_name}.npz", allow_pickle=True)
    channel_name = str(data["channels"][channel_index])
    idx = selected_feature_indices(data["feature_names"], channel_name)
    names = data["feature_names"][idx]
    frames = {
        split: make_frame(data[f"X_{split}"][:, idx], data[f"y_{split}"], names)
        for split in ["train", "val", "test"]
    }
    print(f"Loaded {file_name} | channel {channel_index}: {channel_name} | features: {len(idx)}")
    return frames

def load_file_pooled(file_name, path=SPLIT_PATH):
    """
    Loads a file and pools all channels together as separate samples.
    Strips channel names from features to ensure column alignment.
    """
    file_path = path / f"{file_name}.npz"
    data = np.load(file_path, allow_pickle=True)
    
    orig_train_size = len(data["X_train"])
    channels = data["channels"]
    feature_names = data["feature_names"]
    
    # Pre-identify the base feature names (removing the "Channel_" prefix)
    # first channel's features as a template to select the others
    first_ch = str(channels[0])
    base_names = [name.replace(f"{first_ch}_", "") for name in feature_names if name.startswith(f"{first_ch}_")]
    
    combined_frames = {"train": [], "val": [], "test": []}
    
    for ch_name in channels:
        ch_name = str(ch_name)
        # find indices for this specific channel
        idx = [i for i, name in enumerate(feature_names) if name.startswith(f"{ch_name}_")]
        
        if len(idx) == 0: continue
            
        for split in ["train", "val", "test"]:
            X_split = data[f"X_{split}"][:, idx]
            y_split = data[f"y_{split}"]
            
            # Create frame with generic names (e.g., 'delta_pwr' instead of 'FP1-F7_delta_pwr')
            df = pd.DataFrame(X_split, columns=base_names)
            df["is_seizure"] = np.asarray(y_split).astype(int)
            # keep track of which channel the sample came from
            # df["original_channel"] = ch_name 
            
            combined_frames[split].append(df)
            
    # stack all channels for each split
    final_splits = {
        split: pd.concat(combined_frames[split], ignore_index=True) 
        for split in ["train", "val", "test"]
    }
    
    print(f"Loaded {file_name} (original size = {orig_train_size})| Pooled {len(channels)} channels | New Train Size: {len(final_splits['train'])}")
    return final_splits



def load_models(model_path):
    """
    Load the models in a folder path, and return a list of all available models
    """
    models_list = []

    for file in os.listdir(model_path):
        if file.endswith((".pkl", ".joblib", ".model")):
            file_path = os.path.join(model_path, file)

            try:
                model = joblib.load(file_path)
                model.original_filename = file 
                models_list.append(model)
                print(f"Modello caricato: {file}")

            except Exception as e:
                print(f"\nUnable to load the model {file}: {e}")

    if len(models_list) < 1:
        print(f"\nNo models found in path {model_path}")

    return models_list
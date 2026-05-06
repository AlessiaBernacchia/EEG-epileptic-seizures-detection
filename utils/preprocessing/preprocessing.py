import mne
import os
from glob import glob


def load_files(base_path, exclude_files):

        
    patients = [p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p)) and p.startswith("chb")]

    files = {}
    for patient in patients:
        print(f"\nElaborating patient {patient}")
        current_dir = os.path.join(base_path, patient)
            
        edf_files = sorted(glob(os.path.join(current_dir, "*.edf")))
        
        for file in edf_files:

            name = os.path.basename(file)
            
            name_without_ext = os.path.splitext(name)[0]
            
            if name_without_ext in exclude_files:
                print(f"\nSkipping file {name_without_ext}")
                continue
            
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
            
            files[name_without_ext] = raw
            
        print(f"\nFinish to elaborate patient {patient}")
        
    return files

def data_dict(data_filt, ch_names, scaler):
    """
    scale the normal and filtered data, and return two dictionaries 
    containing the channels.
    
    Args:
        data (Array): array containing the standard data
        
        data_filt (Array): array containing the filtered data
        
        scaler: type of scaler used to scale the both normal and filtered data
        
        ch_names (List): list containing the channels present in the data
        
        
    Returns:
                        
        data_ch_filt (Dict): dictionary of filtered data scaled
    """
    

    # Scale filtered data
    data_filt_scaled = scaler.fit_transform(data_filt.T).T

    # Dictionary of the filtered scaled data
    ch_filt_dict = dict(zip(ch_names, data_filt_scaled))
    
    data_filt_dict = {name: data for name, data in ch_filt_dict.items() if name != "-" and "T8-P8" not in name}
    
    return data_filt_dict
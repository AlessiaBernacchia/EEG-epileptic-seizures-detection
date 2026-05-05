import os
import re
import requests
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm

def collect_info(target_dir=None):
    """
    Download summary and general info files from PhysioNet CHB-MIT database using wfdb.
    """
    # 1. define target directory for downloading data
    if target_dir is None:
        # use the default path relative to the project structure
        base_dir = Path(__file__).resolve().parent
        target_dir = base_dir.parent.parent / "data" / "raw" / "info"
    else:
        # if defined, use the provided
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Target directory: {target_dir}")

    # The CHB-MIT database slug on PhysioNet
    db_name = 'chbmit'
    
    try:
        # 2. download SUBJECT-INFO
        print("[-] Downloading SUBJECT-INFO...")
        wfdb.dl_files(db_name, str(target_dir), ['SUBJECT-INFO'])
        
        # 3. download all summary files (chb01-summary.txt, etc.)
        # We fetch the list of folders/records and append '-summary.txt' to each
        print("[-] Fetching record list from PhysioNet...")
        records = wfdb.get_record_list(db_name)
        
        # summary files are usually stored as 'chb01/chb01-summary.txt'
        folders = sorted(list(set([r.split('/')[0] for r in records])))
        summary_files = [f"{folder}/{folder}-summary.txt" for folder in folders]
        
        print(f"[-] Downloading {len(summary_files)} summary files...")
        # Removed keep_full_path to ensure compatibility with older/newer wfdb versions
        wfdb.dl_files(db_name, str(target_dir), summary_files)
        
        print(f"\n[OK] Download completed successfully in: {target_dir.absolute()}")
    except Exception as e:
        print(f"[ERROR] An error occurred while downloading: {e}")

def collect_all_rec_of(patient_id, target_dir=None):
    """
    Download .edf recordings for a specific patient (e.g., patient_id=4 or '04').
    """
    # 1. format the patient ID (e.g., '04')
    p_id = str(patient_id).zfill(2)
    
    # 2. define target directory
    if target_dir is None:
        base_dir = Path(__file__).resolve().parent
        target_dir = base_dir.parent.parent / "data" / "raw" / "records" / f"chb{p_id}"
    else:
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Target directory: {target_dir}")
    
    try:
        # 3. fetch record list
        print(f"[-] Fetching record list for patient {p_id}...")
        all_records = wfdb.get_record_list('chbmit')
        
        # 4. filter records for the specific patient
        patient_records = [
            r for r in all_records 
            if r.startswith(f"chb{p_id}/") or r.startswith(f"{p_id}/")
        ]
        
        if not patient_records:
            print(f"[!] No records found for patient {p_id}.")
            return

        # 5. download .edf files
        # FIX: Only add .edf if it's not already part of the record name
        edf_files = []
        for r in patient_records:
            if r.endswith('.edf'):
                edf_files.append(r)
            else:
                edf_files.append(f"{r}.edf")
        
        print(f"[-] Downloading {len(edf_files)} .edf files for chb{p_id}...")
        wfdb.dl_files('chbmit', str(target_dir), edf_files)
        
        print(f"\n[OK] Download of chb{p_id} recordings completed.")
    except Exception as e:
        print(f"[ERROR] Failed to download recordings: {e}")



def collect_rec_of(patient_id, target_dir=None, info_dir=None, non_seizure_limit=2):
    """
    Optimized download of .edf recordings for a specific patient.
    It parses the summary file first to download all seizure files 
    and a limited number of non-seizure files using requests for speed.
    """
    # 1. Format the patient ID (e.g., '04')
    p_id = str(patient_id).zfill(2)
    p_name = f"chb{p_id}"
    
    # 2. Define directories
    base_dir = Path(__file__).resolve().parent
    if target_dir is None:
        target_dir = base_dir.parent.parent / "data" / "raw" / "records" / p_name
    else:
        target_dir = Path(target_dir)
    
    if info_dir is None:
        info_dir = base_dir.parent.parent / "data" / "raw" / "info" 
    else:
        info_dir = Path(info_dir)

    # 3. Locate summary file
    summary_pattern = f"**/{p_name}-summary.txt"
    summary_matches = list(info_dir.glob(summary_pattern))
    
    if not summary_matches:
        print(f"[!] Summary file for {p_name} not found in {info_dir}. Please run collect_info() first.")
        return
    
    summary_path = summary_matches[0]

    with open(summary_path, 'r') as f:
        content = f.read()

    # 4. Parse summary to separate files with and without seizures
    file_blocks = content.split("File Name: ")
    seizure_files = []
    normal_files = []

    for block in file_blocks[1:]:  # Skip header
        file_line = block.split('\n')[0].strip()
        file_name = Path(file_line).name 
        
        num_seizures_match = re.search(r"Number of Seizures in File: (\d+)", block)
        
        if num_seizures_match:
            num_seizures = int(num_seizures_match.group(1))
            if num_seizures > 0:
                seizure_files.append(file_name)
            else:
                normal_files.append(file_name)

    # 5. Apply selection logic
    selected_normal = normal_files[:non_seizure_limit]
    files_to_download = seizure_files + selected_normal
    
    if not files_to_download:
        print(f"[!] No files identified for patient {p_name}.")
        return

    # 6. Targeted Download using Requests and TQDM
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Patient {p_name}: Found {len(seizure_files)} seizure files and {len(normal_files)} normal files.")
    print(f"[-] Selected {len(files_to_download)} files for download (Limit non-seizure: {non_seizure_limit}).")

    for f in files_to_download:
        file_url = f"https://physionet.org/files/chbmit/1.0.0/{p_name}/{f}"
        save_path = target_dir / f
        
        if save_path.exists():
            print(f"[!] File {f} already exists. Skipping...")
            continue

        # Retry logic: try up to 3 times for each file
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Added timeout of 30 seconds to avoid hanging forever
                response = requests.get(file_url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(save_path, 'wb') as file, tqdm(
                    desc=f"Downloading {f}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024 * 1024): 
                        size = file.write(data)
                        bar.update(size)
                
                # If download is successful, break the retry loop
                break 

            except (requests.exceptions.RequestException, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"[!] Connection error for {f}. Retrying ({attempt + 2}/{max_retries})...")
                else:
                    print(f"[ERROR] Failed to download {f} after {max_retries} attempts: {e}")

def make_info_df(info_dir=None, save_csv=False, csv_path=None):
    """
    Make a DataFrame containing information about the collected data by parsing summary files.
    """
    if info_dir is None:
        base_dir = Path(__file__).resolve().parent
        info_dir = base_dir.parent.parent / "data" / "raw" / "info"
    else:
        info_dir = Path(info_dir)

    # 1. Read SUBJECT-INFO for demographics (Sex, Age)
    sub_info_path = info_dir / "SUBJECT-INFO"
    if not sub_info_path.exists():
        # Searching recursively in case wfdb nested it
        sub_info_path = next(info_dir.glob("**/SUBJECT-INFO"), None)

    if sub_info_path:
        df_subjects = pd.read_csv(sub_info_path, sep='\t', names=['Subject', 'Sex', 'Age'])
        df_subjects['Subject'] = df_subjects['Subject'].str.lower()
    else:
        print("[WARNING] SUBJECT-INFO not found. Demographics will be missing.")
        df_subjects = pd.DataFrame(columns=['Subject', 'Age', 'Sex'])

    data_list = []

    # 2. Parse every summary.txt file recursively (**/)
    summary_files = list(info_dir.glob("**/chb*-summary.txt"))
    
    for file_path in summary_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract patient ID (e.g., chb01)
        patient_name = file_path.name.split('-')[0]
        
        # 3. Extract Sampling Frequency
        freq_match = re.search(r"Sampling Rate: (\d+) Hz", content)
        freq = int(freq_match.group(1)) if freq_match else 256
        
        # 4. Extract Recording Durations
        # Finds differences between 'File End Time' and 'File Start Time'
        start_times = re.findall(r"File Start Time: (\d{2}:\d{2}:\d{2})", content)
        end_times = re.findall(r"File End Time: (\d{2}:\d{2}:\d{2})", content)
        
        # Note: In CHB-MIT, total seconds are often easier to parse via seizure markers
        # For simplicity in this summary, we track time windows of files
        file_names = re.findall(r"File Name: (.*)", content)
        
        # 5. Extract Seizure Durations
        # Regex to find Start and End times in seconds
        s_starts = re.findall(r"Seizure \d* ?Start Time: (\d+) seconds", content)
        s_ends = re.findall(r"Seizure \d* ?End Time: (\d+) seconds", content)
        
        seizure_durations = [int(e) - int(s) for s, e in zip(s_starts, s_ends)]
        
        # 6. Aggregate data for the subject
        data_list.append({
            'Subject': patient_name,
            'Sampling_Freq': freq,
            'Num_Seizures': len(seizure_durations),
            'Seizure_Durations': seizure_durations,
            'Avg_Seizure_Dur': pd.Series(seizure_durations).mean() if seizure_durations else 0,
            'Min_Seizure_Dur': pd.Series(seizure_durations).min() if seizure_durations else 0,
            'Max_Seizure_Dur': pd.Series(seizure_durations).max() if seizure_durations else 0,
            'Num_Files': len(file_names)
        })

    df_stats = pd.DataFrame(data_list)
    
    # 7. Merge with demographics
    final_df = pd.merge(df_stats, df_subjects, on='Subject', how='left')

    if save_csv:
        if csv_path is None:
            csv_path = info_dir / "collected_info.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"[OK] DataFrame saved to {csv_path}")
    
    return final_df

    """
    Show the collected information of a specific patient.
    """
    p_id = f"chb{str(patient_id).zfill(2)}"
    patient_row = df[df['Subject'] == p_id]

    if patient_row.empty:
        print(f"[!] No data found for subject {p_id}")
        return

    row = patient_row.iloc[0]
    print(f"\n{'='*30}")
    print(f"REPORT FOR SUBJECT: {p_id.upper()}")
    print(f"{'='*30}")
    print(f"Demographics: Age {row['Age']} | Sex {row['Sex']}")
    print(f"Recording:    {row['Num_Files']} files | {row['Sampling_Freq']} Hz")
    print(f"Seizures:     {row['Num_Seizures']} total")
    if row['Num_Seizures'] > 0:
        print(f"Durations:    Avg: {row['Avg_Seizure_Dur']:.2f}s | Min: {row['Min_Seizure_Dur']}s | Max: {row['Max_Seizure_Dur']}s")
        print(f"List (s):     {row['Seizure_Durations']}")
    print(f"{'='*30}\n")
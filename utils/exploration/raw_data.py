import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import mne
import warnings

def show_info(df, save=False, save_path=None, name="global_info_overview.png"):
    """
    Advanced dashboard for EEG dataset analysis.
    Features: Pie Chart (Absolute counts), Age by Sex (Stacked), 
    Seizure vs Normal file distribution, and Per-patient Seizure Durations.
    """
    if df is None or df.empty:
        print("[ERROR] DataFrame is empty.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    
    # Custom Palette
    sex_colors = {"F": "#FF1493", "M": "#1E90FF"} # Pink Fluo and Deep Sky Blue

    # 1. Sex Distribution (Pie Chart with ABSOLUTE numbers)
    sex_counts = df['sex'].value_counts()
    def absolute_value(val):
        # val is the percentage, we convert it back to the absolute number
        return int(np.round(val/100.*sex_counts.sum()))

    axes[0, 0].pie(sex_counts, labels=sex_counts.index, autopct=absolute_value, 
                   colors=[sex_colors[k] for k in sex_counts.index], startangle=140,
                   textprops={'fontsize': 14, 'weight': 'bold'})
    axes[0, 0].set_title("Total Subjects by Sex (Count)", fontsize=15)

    # 2. Age Distribution (Ordered by Age, Stacked by Sex)
    df_age = df.copy()
    df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')
    df_age = df_age.dropna(subset=['age']).sort_values('age')
    
    sns.histplot(data=df_age, x='age', hue='sex', multiple="stack", 
                 ax=axes[0, 1], palette=sex_colors, discrete=True, edgecolor='white', alpha=0.8)
    axes[0, 1].set_title("Age Distribution (Stacked by Sex)", fontsize=15)
    # Set x-ticks for every year present in the dataset
    age_range = range(int(df_age['age'].min()), int(df_age['age'].max()) + 1)
    axes[0, 1].set_xticks(age_range)

    # 3. Records per Subject (Stacked: Seizure Files vs Normal Files)
    df_sorted = df.sort_values('subject')
    
    # Plotting Total Files first (as the background/Normal part)
    sns.barplot(data=df_sorted, x='subject', y='num_files', color='#1E90FF', label='normal_files', ax=axes[0, 2])
    # Overlaying the Seizure Files on top (Red part)
    sns.barplot(data=df_sorted, x='subject', y='num_seizures', color='#FF0000', label='seizure_files', ax=axes[0, 2])
    
    axes[0, 2].set_title("File Composition (Red = Seizure Files)", fontsize=15)
    axes[0, 2].tick_params(axis='x', rotation=70)
    axes[0, 2].legend()

    # 4. Seizure Duration Boxplot
    # We explode the list to get individual seizure durations for the plot
    all_seizures = df.explode('seizure_durations')
    sns.boxplot(y=all_seizures['seizure_durations'], ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title("Individual Seizure Durations (s)")

    # 5. Number of Seizures Boxplot
    sns.boxplot(y=df['num_seizures'], ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title("Number of Seizures per Subject")

    # 6. Heatmap of correlation
    # Mapping Sex to numeric for correlation
    corr_df = df.copy()
    corr_df['sex_code'] = corr_df['sex'].map({'M': 1, 'F': 0})
    numeric_cols = ['age', 'num_seizures', 'avg_seizure_dur', 'sex_code']
    sns.heatmap(corr_df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1, 2])
    axes[1, 2].set_title("Correlation Heatmap")

    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path / name)

    plt.tight_layout()
    plt.show()

def show_seizure_analysis(df, show_global=False, log_scale=True, save=False, save_path=None, name="global_seizure_info_overview.png"):
    """
    Detailed seizure analysis dashboard.
    Top: Seizure counts per patient (and optionally Global total).
    Bottom: Seizure durations in Log Scale, colored by Sex.
    """
    if df is None or df.empty:
        print("[ERROR] DataFrame is empty.")
        return

    # 1. Data Preparation
    temp_df = df.copy()
    temp_df['sex'] = temp_df['sex'].fillna('U')
    
    all_seizures = temp_df.explode('seizure_durations')
    all_seizures['seizure_durations'] = pd.to_numeric(all_seizures['seizure_durations'])
    
    subjects_df = temp_df.sort_values('subject')
    subjects = subjects_df['subject'].tolist()
    counts = subjects_df['num_seizures'].tolist()
    
    # Sex-based colors
    sex_colors_map = {"F": "#FF1493", "M": "#1E90FF", "U": "#808080"}
    subject_sex_colors = [sex_colors_map.get(s, "#808080") for s in subjects_df['sex']]
    
    # Data for boxplot
    plot_data_dur = [all_seizures[all_seizures['subject'] == s]['seizure_durations'].dropna() for s in subjects]

    # 2. Add Global Data if requested
    labels = subjects.copy()
    if show_global:
        labels.append("GLOBAL")
        counts.append(sum(counts)) # Total seizures
        plot_data_dur.append(all_seizures['seizure_durations'].dropna()) # All durations
        subject_sex_colors.append("#FFD700") # Gold color for Global box

    # Numeric positions for alignment
    x_pos = np.arange(len(labels))

    # 3. Setup Figure
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(24, 14), sharex=True, 
                                           gridspec_kw={'height_ratios': [1, 2]})
    

    # TOP PLOT: Number of Seizures
    # Highlight global bar with a darker color if present
    bar_colors = ['#FF0000' if l != "GLOBAL" else '#B22222' for l in labels]
    ax_top.bar(x_pos, counts, color=bar_colors, edgecolor='black', alpha=0.8, width=0.7)
    
    for i, v in enumerate(counts):
        ax_top.text(i, v + 0.2, str(v), ha='center', fontweight='bold', fontsize=12)
        
    ax_top.set_title(f"Seizure Analysis {'(including Global)' if show_global else ''}", fontsize=20, pad=25)
    ax_top.set_ylabel("Count (Seizures)", fontsize=14)
    ax_top.grid(axis='y', linestyle='--', alpha=0.5)

    # BOTTOM PLOT: Seizure Durations (Log Scale if desired)
    bplot = ax_bottom.boxplot(plot_data_dur, positions=x_pos, patch_artist=True, 
                              widths=0.6, medianprops={'color': 'black', 'linewidth': 2})

    # Apply colors to boxes
    for patch, color in zip(bplot['boxes'], subject_sex_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Log Scale and Labels
    if log_scale:
        ax_bottom.set_yscale('log')
    ax_bottom.set_ylabel("Duration (seconds) - Log Scale", fontsize=14)
    ax_bottom.set_xlabel("Subjects", fontsize=14)
    
    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax_bottom.grid(True, which="both", axis='y', linestyle='--', alpha=0.4)

    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path / name)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05) 
    plt.show()

def show_info_of_patient(df, patient_id):
    """
    Display detailed statistics and clinical profile for a specific patient.
    """
    # 1. Format and find the subject
    p_id = f"chb{str(patient_id).zfill(2)}"
    patient_row = df[df['subject'] == p_id]

    if patient_row.empty:
        print(f"[!] No data found for subject {p_id}. Ensure make_info_df() was run.")
        return

    row = patient_row.iloc[0]
    
    # 2. Calculated Metrics
    seizure_durations = row['seizure_durations']
    total_seizure_time = sum(seizure_durations)
    
    # Approximate total recording time (Num_Files * ~1 hour average)
    # Note: For more precision, we would need to sum the actual durations from the summary
    total_files = row['num_files']
    avg_seizure_dur = row['avg_seizure_dur']

    # CHB-MIT files are roughly 1 hour (3600s) each
    est_total_time_s = total_files * 3600
    seizure_burden_pct = (total_seizure_time / est_total_time_s) * 100 if est_total_time_s > 0 else 0
    
    # 3. Print Report
    print(f"\n" + "="*50)
    print(f" CLINICAL PROFILE: {p_id.upper()}")
    print("="*50)
    
    print(f" DEMOGRAPHICS")
    print(f" ├─ Age: {row['age']}")
    print(f" └─ Sex: {row['sex']}")
    
    print(f"\n DATASET STATS")
    print(f" ├─ Total EEG Records:  {total_files}")
    print(f" ├─ Est. Total Time:    ~{est_total_time_s} hours")
    print(f" └─ Sampling Frequency: {row['sampling_freq']} Hz")
    
    print(f"\n SEIZURE ANALYSIS")
    print(f" ├─ Total Seizures:     {row['num_seizures']}")
    if row['num_seizures'] > 0:
        print(f" ├─ Total Seizure Time: {total_seizure_time}s (~{total_seizure_time/60:.2f} min)")
        print(f" ├─ Average Duration:   {avg_seizure_dur:.2f}s")
        print(f" ├─ Min/Max Duration:   {row['min_seizure_dur']}s / {row['max_seizure_dur']}s")
        print(f" └─ Seizure Burden:     {total_seizure_time/3600:.3f} hours of ictal activity")
        print(f"                        {seizure_burden_pct:.2f}% of total recording")
    else:
        print(f" └─ Seizure activity:   None detected in summary")
    
    print("="*50 + "\n")

    # 4. Quick Visualization of Seizure Durations for this patient
    if row['num_seizures'] > 1:
        plt.figure(figsize=(10, 4))
        sns.histplot(seizure_durations, bins=10, kde=True, color='salmon')
        plt.title(f"Distribution of Seizure Durations - Subject {p_id.upper()}")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.show()

def plot_channel_availability(patient_id, records_dir, cleaned=True, remove_warnings=True):
    """
    Heatmap of channel availability from the records.
    Optionally silences MNE/Python warnings and cleans ghost/duplicate channels.
    """
    if remove_warnings:
        # Silence MNE internal logging
        mne.set_log_level('ERROR')
    
    p_id = f"chb{str(patient_id).zfill(2)}"
    path = Path(records_dir) / p_id
    edf_files = sorted(list(path.glob("*.edf")))
    
    if not edf_files:
        print(f"[!] No .edf files found in {path}")
        return

    file_channels = {}
    all_unique_channels = set()
    
    for f in edf_files:
        # Context manager to catch and ignore Python RuntimeWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
            channels = raw.ch_names
        
        if cleaned:
            # 1. Remove ghost channels (dots, dashes, empty strings)
            channels = [ch for ch in channels if ch.strip() not in ['', '-', '.', '--', '---']]
            # 2. Filter out channels MNE auto-renames with numbers (like '--0')
            channels = [ch for ch in channels if not (ch.startswith('--') and ch[2:].isdigit())]
            # 3. Handle duplicates: keep first occurrence
            seen = set()
            channels = [x for x in channels if not (x in seen or seen.add(x))]
            
        file_channels[f.name] = channels
        all_unique_channels.update(channels)
    
    if remove_warnings:
        # Reset log level to default after the loop
        mne.set_log_level('INFO')
    
    all_unique_channels = sorted(list(all_unique_channels))
    matrix = pd.DataFrame(0, index=all_unique_channels, columns=[f.name for f in edf_files])
    
    for f_name, channels in file_channels.items():
        matrix.loc[channels, f_name] = 1

    plt.figure(figsize=(16, 10))
    sns.heatmap(matrix, cmap=["#f0f0f0", "#1E90FF"], cbar=False, linewidths=.3, linecolor='white')
    
    status = " (Cleaned & Silent)" if cleaned else ""
    plt.title(f"Channel Consistency: {p_id.upper()}{status}")
    plt.xlabel("EDF Files")
    plt.ylabel("EEG Channels")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_channel_availability_from_summary(patient_id, info_dir, filter_downloaded=True, raw_records_path=None, verbose=False,):
    """
    Heatmap of channels from summary.txt. 
    0: Absent (Grey), 1: Present (Blue), 2: Seizure File (Red).
    """
    # helper normalization functions
    def normalize_filename(name):
        return str(name).strip()

    def normalize_channel(name):
        return str(name).strip().upper()

    # 1. Define paths
    info_dir = Path(info_dir)
    p_id = f"chb{int(patient_id):02d}"
    summary_path = info_dir / p_id / f"{p_id}-summary.txt"
    if not summary_path.exists():
        summary_path = info_dir / f"{p_id}-summary.txt"
        if not summary_path.exists():
            print(f"[ERROR] Summary file not found for {p_id} in {info_dir}")
            return

    raw_records_path = Path(raw_records_path) if raw_records_path is not None else Path("./data/raw/records")

    if verbose:
        print(f"[INFO] patient={p_id}")
        print(f"[INFO] summary_path={summary_path}")
        print(f"[INFO] raw_records_path={raw_records_path}")
        print(f"[INFO] filter_downloaded={filter_downloaded}")

    with summary_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. Find channels files
    file_channels = {}
    seizure_files = set()
    all_unique_channels = set()
    current_montage = []
    last_f_name = None

    for line in lines:
        # Update current montage
        ch_match = re.search(r"Channel \d+: (\S+)", line)
        if ch_match:
            ch = normalize_channel(ch_match.group(1))
            current_montage.append(ch)

        # Reset montage tracking on header lines
        if "Channels" in line and ("changed" in line.lower() or "in edf" in line.lower()):
            current_montage = []

        # Map file name to current montage
        if "File Name:" in line:
            last_f_name = normalize_filename(line.split(":", 1)[1])
            file_channels[last_f_name] = list(current_montage)
            all_unique_channels.update(current_montage)

        # Detect if the current file has seizures
        if "Number of Seizures in File:" in line and last_f_name:
            num = int(re.search(r"\d+", line).group())
            if num > 0:
                seizure_files.add(last_f_name)

    final_files = list(file_channels.keys())

    # 3. Filter by records downloaded
    if filter_downloaded:
        local_path = raw_records_path / p_id
        if not local_path.exists():
            print(f"[ERROR] Records folder not found: {local_path}")
            return

        downloaded_on_disk = sorted(normalize_filename(f.name) for f in local_path.glob("*.edf"))
        if verbose:
            print(f"[INFO] downloaded files count={len(downloaded_on_disk)}")

        matched_files = [f for f in final_files if f in downloaded_on_disk]
        if verbose:
            print(f"[INFO] matched files count={len(matched_files)}")

        if not matched_files:
            print(f"[ERROR] No matching EDF files found after filtering for {p_id}")
            return

        final_files = matched_files

    # 4. Matrix Construction
    all_unique_channels = sorted(set(all_unique_channels))
    # Remove placeholder channels like '-'
    all_unique_channels = [ch for ch in all_unique_channels if ch.strip() and ch != '-']
    
    matrix = pd.DataFrame(0, index=all_unique_channels, columns=final_files)

    for f_name in final_files:
        val = 2 if f_name in seizure_files else 1
        channels = file_channels.get(f_name, [])
        if channels:
            for ch in channels:
                if ch in matrix.index:
                    matrix.loc[ch, f_name] = val

    if matrix.empty or matrix.shape[1] == 0 or matrix.shape[0] == 0:
        print(f"[ERROR] Matrix is empty for {p_id}")
        return

    matrix_filtered = matrix.loc[(matrix != 0).any(axis=1)]
    if matrix_filtered.empty:
        print(f"[ERROR] After removing empty rows, matrix is empty for {p_id}")
        return

    if verbose:
        print(f"[INFO] final matrix shape={matrix_filtered.shape}")
        print(f"[INFO] total nonzero entries={(matrix_filtered != 0).sum().sum()}")

    # 5. Plot with custom 3-color palette
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap( matrix_filtered, cmap=["#f0f0f0", "#1E90FF", "#FF0000"], cbar=False, linewidths=.1, linecolor='white', ax=ax, )

    ax.set_title(f"Channel Consistency: {p_id.upper()} (Red = Seizure)", fontsize=16)
    ax.set_xlabel("Files", fontsize=12)
    ax.set_ylabel("EEG Channels", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

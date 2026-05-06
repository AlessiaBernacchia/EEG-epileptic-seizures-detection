import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_info(df):
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

    plt.tight_layout()
    plt.show()

def show_seizure_analysis(df, show_global=False, log_scale=True):
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
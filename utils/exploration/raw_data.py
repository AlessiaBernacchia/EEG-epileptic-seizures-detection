import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

def show_info(df):
    """
    Show the collected information with plots.
    """
    if df is None or df.empty:
        print("[ERROR] DataFrame is empty.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # 1. Male vs Female
    sns.countplot(data=df, x='Sex', ax=axes[0, 0], palette='viridis', hue='Sex', legend=False)
    axes[0, 0].set_title("Sex Distribution")

    # 2. Age Distribution
    sns.histplot(data=df, x='Age', kde=True, ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title("Age Distribution")

    # 3. Number of Files per Subject
    sns.barplot(data=df, x='Subject', y='Num_Files', ax=axes[0, 2])
    axes[0, 2].set_title("Records per Subject")
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. Seizure Duration Boxplot
    # We explode the list to get individual seizure durations for the plot
    all_seizures = df.explode('Seizure_Durations')
    sns.boxplot(y=all_seizures['Seizure_Durations'], ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title("Individual Seizure Durations (s)")

    # 5. Number of Seizures Boxplot
    sns.boxplot(y=df['Num_Seizures'], ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title("Number of Seizures per Subject")

    # 6. Heatmap of correlation
    # Mapping Sex to numeric for correlation
    corr_df = df.copy()
    corr_df['Sex_Code'] = corr_df['Sex'].map({'M': 1, 'F': 0})
    numeric_cols = ['Age', 'Num_Seizures', 'Avg_Seizure_Dur', 'Sex_Code']
    sns.heatmap(corr_df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=axes[1, 2])
    axes[1, 2].set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()

def show_info_of_patient(df, patient_id):
    """
    Display detailed statistics and clinical profile for a specific patient.
    """
    # 1. Format and find the subject
    p_id = f"chb{str(patient_id).zfill(2)}"
    patient_row = df[df['Subject'] == p_id]

    if patient_row.empty:
        print(f"[!] No data found for subject {p_id}. Ensure make_info_df() was run.")
        return

    row = patient_row.iloc[0]
    
    # 2. Calculated Metrics
    seizure_durations = row['Seizure_Durations']
    total_seizure_time = sum(seizure_durations)
    
    # Approximate total recording time (Num_Files * ~1 hour average)
    # Note: For more precision, we would need to sum the actual durations from the summary
    total_files = row['Num_Files']
    avg_seizure_dur = row['Avg_Seizure_Dur']
    
    # 3. Print Report
    print(f"\n" + "="*50)
    print(f" CLINICAL PROFILE: {p_id.upper()}")
    print("="*50)
    
    print(f" DEMOGRAPHICS")
    print(f" ├─ Age: {row['Age']}")
    print(f" └─ Sex: {row['Sex']}")
    
    print(f"\n DATASET STATS")
    print(f" ├─ Total EEG Records:  {total_files}")
    print(f" └─ Sampling Frequency: {row['Sampling_Freq']} Hz")
    
    print(f"\n SEIZURE ANALYSIS")
    print(f" ├─ Total Seizures:     {row['Num_Seizures']}")
    if row['Num_Seizures'] > 0:
        print(f" ├─ Total Seizure Time: {total_seizure_time}s (~{total_seizure_time/60:.2f} min)")
        print(f" ├─ Average Duration:   {avg_seizure_dur:.2f}s")
        print(f" ├─ Min/Max Duration:   {row['Min_Seizure_Dur']}s / {row['Max_Seizure_Dur']}s")
        print(f" └─ Seizure Burden:     {total_seizure_time/3600:.3f} hours of ictal activity")
    else:
        print(f" └─ Seizure activity:   None detected in summary")
    
    print("="*50 + "\n")

    # 4. Quick Visualization of Seizure Durations for this patient
    if row['Num_Seizures'] > 1:
        plt.figure(figsize=(10, 4))
        sns.histplot(seizure_durations, bins=10, kde=True, color='salmon')
        plt.title(f"Distribution of Seizure Durations - Subject {p_id.upper()}")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.show()
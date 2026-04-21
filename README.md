# EEG-epileptic-seizures-detection
The goal of this project is to develop methods to detect epileptic seizures from electroencephalographic (EEG) signals. The project involves analyzing brain activity recordings to distinguish between normal and status epilepticus and exploring how temporal patterns in EEG signals can be used for reliable and interpretable classification.

## Dataset
The dataset used in this project is the "Epileptic Seizure Recognition" dataset, which contains EEG recordings from 23 patients. Each recording is labeled as either normal or status epilepticus, providing a basis for training and evaluating classification models.

- [dataset source link](https://physionet.org/content/chbmit/1.0.0/)
- [paper describing the dataset](https://physionet.org/content/chbmit/1.0.0/shoeb-icml-2010.pdf)
- size: 23 patients, 24 channels, 1-second segments, sampled at 256 Hz

## Methodology
The project will involve the following steps:
1. **Data exploration and preprocessing**: Load and inspect the EEG recordings and associated labels, handle noise and artifacts (e.g., filtering, normalization), and segment the signal into time windows suitable for analysis.
2. **Feature extraction / representation**: Extract time-domain features (e.g., amplitude statistics, variance), frequency-domain or time-frequency features (e.g., power spectral density, spectrograms), and optionally apply dimensionality reduction techniques (e.g., PCA) or data-driven representations (e.g., autoencoders).
3. **Model development**: Implement baseline classifiers (e.g., logistic regression, k-nearest neighbors, Naive Bayes) and explore more advanced models such as time-series models (e.g., Hidden Markov Models) and deep learning approaches (e.g., CNNs on spectrograms, recurrent neural networks).
4. **Model evaluation**: Split data into training and test sets, evaluate using metrics such as accuracy, confusion matrix, ROC curves, and AUC, and analyze false positives/false negatives and their implications.
5. **Extension (optional)**: Explore early seizure detection (prediction before onset), patient-specific vs. general models, and interpretability of learned features.

## Setup Environment and Data
To set up the environment and prepare the data for analysis, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AlessiaBernacchia/EEG-epileptic-seizures-detection.git
   ```
2. **Navigate to the project directory**:
   ```bash
    cd EEG-epileptic-seizures-detection
    ```
3. **Install environment with required dependencies**:
   ```bash
    conda env create --file environment.yml
    ```
4. **Activate the environment**:
    ```bash
     conda activate eeg-seizure-detection
     ```
5. **Download the dataset**:
   - Download the "Epileptic Seizure Recognition" dataset from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and place the data files in the `data/` directory of the project.
6. **Run the data preprocessing script**:
   ```bash
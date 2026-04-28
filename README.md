# EEG-epileptic-seizures-detection
The goal of this project is to develop methods to detect epileptic seizures from electroencephalographic (EEG) signals. The project involves analyzing brain activity recordings to distinguish between normal and status epilepticus and exploring how temporal patterns in EEG signals can be used for reliable and interpretable classification.

## Dataset
The dataset used in this project is the "Epileptic Seizure Recognition" dataset, which contains EEG recordings from 23 patients (and ocassionally other signals that we will exclude from analysis). Each recording is labeled as either normal or status epilepticus, providing a basis for training and evaluating classification models.

- providers: Children's Hospital Boston and the Massachusetts Institute of Technology (MIT) 
- [dataset source link](https://physionet.org/content/chbmit/1.0.0/)
- [paper describing the dataset](https://physionet.org/content/chbmit/1.0.0/shoeb-icml-2010.pdf)
- size: 23 patients, 23 channels, about 4-hours segments, sampled at 256 Hz

(the 24th patient `chb24` was added to this collection in December 2010, and is not currently included in `SUBJECT-INFO`)

**Highlights of the providers**:
- Protected health information (PHI) in the original .edf files has been replaced with surrogate identifiers, so we cannot link recordings to specific patients or demographics.
- Dates in the original .edf files have been replaced by surrogate dates that do not correspond to actual recording dates, so we cannot analyze temporal patterns across recordings based on date information.
- The dataset is imbalanced, with more normal recordings than seizure recordings.
- Gaps in the recordings are 10 seconds or less, which may affect the continuity of the data and require careful handling during preprocessing.
- The beginning ([) and end (]) of each seizure is annotated in the `.seizure` annotation files that accompany each of the files listed in `RECORDS-WITH-SEIZURES`.

> **Note**: The dataset is large and may require significant computational resources for processing and analysis. We will focus on a subset of the data to ensure that we can complete the project within the given deadline.

## Methodology
The project will involve the following steps:
1. **Data exploration and preprocessing**: Load and inspect the EEG recordings and associated labels, handle noise and artifacts (e.g., filtering, normalization), and segment the signal into time windows suitable for analysis.
2. **Feature extraction / representation**: Extract time-domain features (e.g., amplitude statistics, variance), frequency-domain or time-frequency features (e.g., power spectral density, spectrograms), and optionally apply dimensionality reduction techniques (e.g., PCA) or data-driven representations (e.g., autoencoders).
3. **Model development**: Implement baseline classifiers (e.g., logistic regression, k-nearest neighbors, Naive Bayes) and explore more advanced models such as time-series models (e.g., Hidden Markov Models) and deep learning approaches (e.g., CNNs on spectrograms, recurrent neural networks).
4. **Model evaluation**: Split data into training and test sets, evaluate using metrics such as accuracy, confusion matrix, ROC curves, and AUC, and analyze false positives/false negatives and their implications.
5. **Extension (optional)**: Explore early seizure detection (prediction before onset), patient-specific vs. general models, and interpretability of learned features.

## Pipeline to follow and Documentation
The project will be organized into a clear and modular pipeline, with separate scripts for data preprocessing, feature extraction, model training, and evaluation. Each step will be documented in detail to ensure reproducibility and clarity of the analysis.

1. [**Setup and Environment**](doc/setup_env.md)
2. [**Data Download and Preprocessing**](doc/data_preprocessing.md)
...

## Foldeer structure:

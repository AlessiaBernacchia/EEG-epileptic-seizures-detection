# EEG-epileptic-seizures-detection
The goal of this project is to develop methods to detect epileptic seizures from electroencephalographic (EEG) signals. The project involves analyzing brain activity recordings to distinguish between normal and status epilepticus and exploring how temporal patterns in EEG signals can be used for reliable and interpretable classification.

## Authors & Contributions
- [Bernacchia Alessia](https://github.com/AlessiaBernacchia)
    - data collection
    - preprocessing (cleaning, feature extraction and labelling)
    - cross-subjects comparison
    - report

- [Pioda Tommaso](https://github.com/Thetommigun432)
    - preprocessing (labelling)
    - modelling (baseline models and deep learning models)
    - report

- [Villani Giacomo](https://github.com/DownToTheGround)
    - preprocessing (splitting)
    - interpretability
    - report

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
3. [**Run Models & Implementation Info**](doc/models_implementation.md)
4. [**Models & Cross-subjects Generalization Evaluation**](doc/cross_subjects_eval.md)
5. [**Interpretability**](doc/interpretability.md)

## Conclusions

The project demonstrated that while detecting an active seizure (**Task 1**) is a relatively stable task for classical machine learning models like KNN, forecasting a seizure before it happens (**Task 2**) remains a complex challenge due to high inter-patient variability.

Key findings include:

* **Baseline Efficiency**: For detection tasks, hyperparameter-tuned classical models (KNN) achieved near-perfect accuracy on stable background activity, making deep learning unnecessary for simple binary classification of ictal vs. inter-ictal states.
* **The "Stability Detection" Paradox**: Models in Task 2 achieved high **Weighted Precision (~0.90)**, proving they are highly reliable at recognizing normal brain activity. However, the low precision in the seizure-specific class highlights the difficulty of distinguishing the subtle 10-minute **pre-ictal** window from standard background noise.
* **Biological Generalization**: Evaluation across subjects revealed that models are highly sensitive to **age and gender**. A model trained on a toddler (chb12) failed to generalize to an adult male (chb04), confirming that brain maturation significantly alters the morphological signature of seizures.
* **Forecasting Sensitivity**: Forecasting (Task 2) proved much more subject-dependent than Detection (Task 1). Even within the same demographic (pediatric), the model struggled to generalize between similar subjects, suggesting that pre-ictal signatures may be unique to an individual's specific neural circuitry.

## Next Steps

* **Subject-Specific Calibration**: Implement a "transfer learning" or fine-tuning approach where a general model is calibrated using a small amount of data from a new patient to improve individual forecasting accuracy.
* **False Positive Reduction**: Integrate a "smoothing" or "persistence" logic (e.g., the model must predict a pre-ictal state for $X$ consecutive windows) to reduce the number of isolated false alarms that cause patient stress.
* **Feature Engineering Expansion**: Explore non-linear features such as **Entropy** or **Phase-Amplitude Coupling**, which may capture the "build-up" to a seizure better than standard power spectral density.
* **Hardware Integration Analysis**: Assess the computational feasibility of deploying these models on edge devices (like wearables) by optimizing the feature extraction step, which is currently the most resource-intensive part of the pipeline.
* **Multi-Subject Training**: Expand the training set from a single subject to a multi-subject pool (clustered by age and gender) to attempt the creation of a more robust "universal" baseline model.

## Folder structure:
```
EEG-epileptic-seizures-detection/
├── data/                       # Git-ignored directory for data storage
│   ├── raw/                    # Original, immutable data
│   │   ├── info/               # Meta-data: SUBJECT-INFO and chbxx-summary.txt files
│   │   └── records/            # selected large .edf brain signal files
│   ├── preprocessed/           # preprocessed files for each subject
│   │   ├── chb04   
│   │   ├── ...
│   │   ...
│   ├── features/               # features extracted for each subject
│   │   ├── chb04   
│   │   ├── ...
│   │   ...
│   ├── labels/                 # features extracted for each subject for each task
│   │   ├── chb04   
│   │   ├── ...
│   │   ...
│   └── splitting/              # final splitting data
│       ├── task_1/             # data splitted for task 01 for each subject
│       └── task_2/             # data splitted for task 02 for each subject
│
├── doc/                        # Project documentation and visualizations
│   ├── src/                    # Exported plots (png) and markdown guides (md)
│   │    ├── exploration/       
│   │    ├── preprocessing/
│   │          ....
│   └── ...                     # all documentation and tutorial markdown files
│ 
├── notebooks/                  # Experimental Jupyter notebooks
│   ├── collection/             # Notebooks for testing data gathering logic
│   ├── preprocessing/          # Notebooks for preprocess data (cleaning, feature extraction, labelling, splitting)   
│   ├── models_task01/          # Notebooks to create, select, compare and interpret models for task 1
│   └── models_task02/          # Notebooks to create, select, compare and interpret models for task 2             
│
├── utils/                      # Core Python package (the logic "engine")
│   ├── collection/             # Modules for downloading and parsing
│   ├── exploration/            # Modules for EDA and visualization (both raw data and preprocessed)
│   ├── data_load /             # Modules for load the data from the splitting (personalized for each task)
│   ├── models/                 # Modules for Models implementation
│   ├── comparison/             # Modules for Models comparison across subjects
│   ├── interpretability/       # Modules for Models interpretability (general and adaptable for all models)
│   ├── model_io.py             # Modules for Models export and import (save and load the models classes)
│   └── __init__.py             # Makes 'utils' an importable package
│
│   
├── saved_models/               # Project saved models (best models for each task)
│   ├── task_1/                 # Exported best models pretrained and hypertuned
│   │    ├── chb<n>_<model-name>_best_model.jonlib
│   │          ....
│   └── task_2/  
│        ├── chb<n>_<model-name>_best_model.jonlib
│            ....
│   
├── .gitignore                  # Instructions on which files Git should ignore (e.g., data/)
├── environment-cpu.yml         # Conda environment specification for cpu dependencies
├── environment-gpu.yml         # Conda environment specification for gpu dependencies
├── pyproject.toml              # Modern build system configuration
├── README.md                   # Main project documentation
└── setup.py                    # Legacy build script for "editable" mode installation
```


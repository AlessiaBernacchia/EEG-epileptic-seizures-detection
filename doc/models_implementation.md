# Model Implementation Guide

## Overview

This document describes all machine learning models implemented for EEG seizure detection, organized by category.

---

## Baseline Classifiers

### 1. Logistic Regression (`LogisticRegressionModel`)

**Description**: Linear classification model using logistic function. Fast and interpretable.

**Key Features**:
- Fast training and inference
- Provides probability estimates
- Good baseline for comparison
- Scales well with RobustScaler

**Usage**:
```python
model = LogisticRegressionModel(model_name='LogisticRegression', max_iter=1000)
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

### 2. Naive Bayes (`NaiveBayesModel`)

**Description**: Probabilistic classifier based on Bayes' theorem with conditional independence assumption.

**Key Features**:
- Fast training
- Good for multi-class problems
- Provides probability estimates
- Works well with Gaussian features

**Usage**:
```python
model = NaiveBayesModel(model_name='NaiveBayes')
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

### 3. Support Vector Machine (`SVMModel`)

**Description**: Discriminative classifier finding optimal separating hyperplane.

**Key Features**:
- Non-linear classification via kernel tricks
- Handles high-dimensional data
- Good generalization
- Requires careful hyperparameter tuning

**Usage**:
```python
model = SVMModel(model_name='SVM', kernel='rbf', C=1.0)
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

### 4. K-Nearest Neighbors (`KNNModel`)

**Description**: Instance-based classifier using distance metrics.

**Key Features**:
- Simple and interpretable
- No training phase
- Memory-intensive for large datasets
- Already implemented in the codebase

**Usage**:
```python
model = KNNModel(model_name='KNN', n_neighbors=5)
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

### 5. Random Forest (`RandomForestModel`)

**Description**: Ensemble of decision trees with random feature selection.

**Key Features**:
- Handles non-linear relationships
- Feature importance ranking
- Robust to outliers
- No feature scaling required

**Usage**:
```python
model = RandomForestModel(model_name='RandomForest', n_estimators=100, max_depth=20)
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

## Gradient Boosting Models

### 6. XGBoost (`XGBModel`)

**Description**: Optimized gradient boosting framework using tree ensembles.

**Key Features**:
- Excellent generalization
- Handles imbalanced data
- GPU acceleration support
- Feature importance extraction

**Usage**:
```python
model = XGBModel(model_name='XGBoost', n_estimators=100, max_depth=7, tree_method='hist')
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

### 7. LightGBM (`LGBModel`)

**Description**: Fast, distributed gradient boosting for classification.

**Key Features**:
- Very fast training
- Low memory usage
- Handles large datasets
- GPU and parallel support

**Usage**:
```python
model = LGBModel(model_name='LightGBM', n_estimators=100, max_depth=7)
X_proc, y = model.preprocess(df_train, is_training=True)
model.train(X_proc, y)
predictions = model.predict(X_test_proc)
```

---

## Advanced Time-Series Models

### 8. Hidden Markov Model (`HiddenMarkovModel`)

**Description**: Probabilistic model for sequential data with hidden states.

**Key Features**:
- Models temporal dependencies
- Two separate models for seizure/normal classes
- Log-likelihood ratio for classification
- Uses hmmlearn library

**Architecture**:
- Training: Separate HMM for seizure and normal patterns
- Prediction: Compare log-likelihoods and compute probability

**Usage**:
```python
model = HiddenMarkovModel(model_name='HMM', n_components=3, covariance_type='full', n_iter=100)
model.train(X_train_proc, y_train)
predictions = model.predict(X_test_proc)
probabilities = model.predict_proba(X_test_proc)
```

**Parameters**:
- `n_components`: Number of hidden states (default: 3)
- `covariance_type`: 'full', 'tied', 'diag', 'spherical'
- `n_iter`: Number of EM iterations (default: 100)

---

## Deep Learning Models

### 9. 1D Convolutional Neural Network (`ConvNetModel`)

**Description**: CNN with multiple parallel convolutions for extracting temporal features.

**Architecture**:
```
Input (1, n_features)
    ↓
[Conv1D (k=3), Conv1D (k=5), Conv1D (k=7)] (parallel)
    ↓ (concatenate)
[Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dropout → Dense(2)]
    ↓
Output
```

**Key Features**:
- Multiple kernel sizes capture different temporal patterns
- Batch normalization and dropout for regularization
- Global average pooling
- Efficient for feature extraction

**Usage**:
```python
model = ConvNetModel(
    model_name='CNN1D',
    input_channels=1,
    num_filters=32,
    dropout_rate=0.3
)
X_train_cnn, y_train = model.preprocess(df_train, is_training=True)
model.train(X_train_cnn, y_train, X_val=X_val, y_val=y_val, epochs=10, batch_size=32)
predictions = model.predict(X_test_cnn, batch_size=32)
```

**Hyperparameters**:
- `num_filters`: Number of filters per convolution (default: 32)
- `dropout_rate`: Dropout probability (default: 0.3)
- `learning_rate`: Adam optimizer LR (default: 1e-3)
- `batch_size`: Training batch size (default: 32)

---

### 10. Recurrent Neural Network - LSTM (`RecurrentNetModel`)

**Description**: Bidirectional LSTM for capturing long-term temporal dependencies.

**Architecture**:
```
Input (1, n_features)
    ↓
Bidirectional LSTM (hidden_size=64, num_layers=2)
    ↓ (take last output)
[Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dropout → Dense(2)]
    ↓
Output
```

**Key Features**:
- Bidirectional processing (forward and backward)
- Multiple stacked LSTM layers
- Captures temporal dependencies over time
- Suitable for sequential pattern recognition

**Usage**:
```python
model = RecurrentNetModel(
    model_name='LSTM',
    hidden_size=64,
    num_layers=2,
    dropout=0.3,
    bidirectional=True
)
X_train_rnn, y_train = model.preprocess(df_train, is_training=True)
model.train(X_train_rnn, y_train, X_val=X_val, y_val=y_val, epochs=10)
predictions = model.predict(X_test_rnn, batch_size=32)
```

**Hyperparameters**:
- `hidden_size`: LSTM hidden dimension (default: 64)
- `num_layers`: Number of LSTM stacks (default: 2)
- `dropout`: Dropout rate (default: 0.3)
- `bidirectional`: Use bidirectional LSTM (default: True)

---

## Data Preprocessing

All models follow a consistent preprocessing pipeline:

### For Multicollinearitz sensitive models
Logistic Regression, Support Vector Machines, K-Nearest Neighbors, Naive Bayes, Hidden Markov Chain models
```markdown
1. Extract feature columns from DataFrame
2. Handle missing values (fill with 0)
3. Scale using RobustScaler
4. Feature Selection (based on correlation)
4. Return features and labels
```
4. feature Selection (to delete multicollinearity if required from the model)

### Tabular Features (Baseline & Boosting)
```markdown
1. Extract feature columns from DataFrame
2. Handle missing values (fill with 0)
3. Scale using RobustScaler
4. Return scaled features and labels
```

### Deep Learning Features
```markdown
1. Extract feature columns
2. Handle missing values
3. Min-Max normalization to [0, 1]
4. Reshape for network input:
   - CNN: (n_samples, 1, n_features)
   - RNN: (n_samples, 1, n_features)
```

---

## Model Comparison

| Model | Type | Training Speed | Interpretability | Performance | Memory |
|-------|------|-----------------|------------------|-------------|--------|
| Logistic Regression | Linear | Fast | High | Baseline | Low |
| Naive Bayes | Probabilistic | Fast | High | Low-Med | Low |
| SVM | Kernel | Medium | Low | Medium | Medium |
| KNN | Instance | None | High | Medium | High |
| Random Forest | Ensemble | Fast | Medium | High | Medium |
| XGBoost | Boosting | Medium | Low | High | Medium |
| LightGBM | Boosting | Fast | Low | High | Low |
| HMM | Time-Series | Slow | Medium | Medium | Low |
| CNN | Deep Learning | Slow | Very Low | Medium-High | Medium |
| LSTM | Deep Learning | Very Slow | Very Low | Medium-High | High |

---

## Training and Evaluation Pipeline

### Complete Training Pipeline
```python
from utils.models.model_classes import LogisticRegressionModel

# Create model
model = LogisticRegressionModel()

# Preprocess
X_train, y_train = model.preprocess(df_train, is_training=True)
X_test, y_test = model.preprocess(df_test, is_training=False)

# Train
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
auc = roc_auc_score(y_test, probabilities)
```

---

## Hyperparameter Tuning

### Quick Start Parameters

**Baseline Models**:
```python
LogisticRegression: max_iter=1000
SVM: kernel='rbf', C=1.0
KNN: n_neighbors=5
RandomForest: n_estimators=100, max_depth=20
```

**Gradient Boosting**:
```python
XGBoost: n_estimators=100, max_depth=7, learning_rate=0.1
LightGBM: n_estimators=100, max_depth=7, learning_rate=0.1
```

**Deep Learning**:
```python
CNN: epochs=10, batch_size=32, learning_rate=1e-3
LSTM: epochs=10, batch_size=32, learning_rate=1e-3, hidden_size=64
```

---

## Notes

1. **Data Format**: All models expect DataFrames with an 'is_seizure' column for labels
2. **Scaling**: Baseline models use RobustScaler automatically
3. **GPU Support**: Deep learning models support GPU via PyTorch CUDA
4. **Early Stopping**: Deep learning models include early stopping with patience
5. **Class Imbalance**: Models handle imbalanced data appropriately

---

## References

- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- hmmlearn: https://github.com/hmmlearn/hmmlearn
- PyTorch: https://pytorch.org/

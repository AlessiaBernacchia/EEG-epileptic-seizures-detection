import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .base import BaseModel

_TORCH_IMPORT_ERROR = None

try:
    import torch
except ImportError:
    _TORCH_IMPORT_ERROR = True
    raise OSError(f"PyTorch is required for deep learning models: {_TORCH_IMPORT_ERROR}")

try:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except (ImportError, OSError):
    nn = None
    DataLoader = None
    TensorDataset = None


LABEL_COLUMNS = ("is_seizure", "label", "target", "y")


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_device(device=None):
    if device is None:
        return get_torch_device()

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return resolved_device


def configure_cuda_runtime():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def loader_kwargs(
    device,
    num_workers: int = 0,
    pin_memory: bool = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    if pin_memory is None:
        pin_memory = getattr(device, "type", None) == "cuda"

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def make_dense_layers(input_size, dense_units, dropout_rate, num_classes):
    layers = []
    previous_size = input_size

    for units in dense_units:
        layers.extend([
            nn.Dropout(dropout_rate),
            nn.Linear(previous_size, units),
            nn.ReLU(),
        ])
        previous_size = units

    layers.extend([
        nn.Dropout(dropout_rate),
        nn.Linear(previous_size, num_classes),
    ])
    return nn.Sequential(*layers)


def label_column(data: pd.DataFrame):
    return next((col for col in LABEL_COLUMNS if col in data.columns), data.columns[-1])


def numeric_feature_frame(data: pd.DataFrame, feature_cols, is_training: bool, model_name: str):
    if feature_cols is None:
        if not is_training:
            raise ValueError(f"{model_name} must be fitted before preprocessing validation/test data")

        target_col = label_column(data)
        candidate_cols = [col for col in data.columns if col != target_col]
        X = data[candidate_cols].select_dtypes(include=[np.number, "bool"]).copy()

        if X.empty:
            raise ValueError(f"[{model_name}] No numeric feature columns found")

        return X, X.columns.tolist()

    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"[{model_name}] Missing feature columns: {missing_cols}")

    X = data[feature_cols].copy()
    return X, feature_cols


def preprocess_tabular_features(model, data: pd.DataFrame, is_training: bool, verbose: bool):
    X, feature_cols = numeric_feature_frame(
        data,
        getattr(model, "feature_cols", None),
        is_training=is_training,
        model_name=model.model_name,
    )
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if is_training or getattr(model, "feature_cols", None) is None:
        model.feature_cols = feature_cols

    if getattr(model, "scaler", None) is None:
        model.scaler = MinMaxScaler()

    if is_training:
        X_scaled = model.scaler.fit_transform(X)
    else:
        X_scaled = model.scaler.transform(X)

    y = data[label_column(data)].to_numpy()

    if verbose:
        print(f"[{model.model_name}] Features shape: {X_scaled.shape}, Labels shape: {y.shape}")
        print("Label distribution:")
        print(pd.Series(y).value_counts(normalize=True))

    return X_scaled.astype(np.float32), y


def resolve_learning_rate(default_learning_rate, kwargs):
    return kwargs.pop("learn_rate", default_learning_rate)


def pos_weight_tensor(y_train, device, class_weights=None, pos_weight=None):
    if pos_weight is None and class_weights is not None:
        if isinstance(class_weights, dict) and 0 in class_weights and 1 in class_weights:
            pos_weight = class_weights[1] / class_weights[0]

    if pos_weight is None:
        y = np.asarray(y_train).astype(int)
        positives = np.sum(y == 1)
        negatives = np.sum(y == 0)
        pos_weight = negatives / positives if positives > 0 else 1.0

    return torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)


def plot_learning_curve(history, title="Learning curve", ax=None, show=True):
    if not history:
        print("No training history available to plot.")
        return ax

    import matplotlib.pyplot as plt

    history_df = pd.DataFrame(history)
    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    metric_cols = [col for col in history_df.columns if col != "epoch"]
    for col in metric_cols:
        ax.plot(history_df["epoch"], history_df[col], marker="o", label=col)

    ax.set_title(title)
    ax.set_xlabel("Epoch" if "epoch" in history_df.columns else "Iteration")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show:
        plt.show()
    return ax


class TorchBinaryClassifierMixin:
    def _plot_learning_curve_after_training(self, show_learning_curve: bool):
        if show_learning_curve:
            self.plot_learning_curve()

    def plot_learning_curve(self, ax=None, show=True):
        return plot_learning_curve(
            self.history,
            title=f"{self.model_name} learning curve",
            ax=ax,
            show=show,
        )

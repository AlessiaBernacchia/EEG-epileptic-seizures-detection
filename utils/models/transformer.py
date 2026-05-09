import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .base import (
    BaseModel,
    _TORCH_IMPORT_ERROR,
    compute_binary_pos_weight,
    df_type_from_name,
    evaluate_classifier_predictions,
    get_torch_device,
    make_tensor_loader,
    slug,
    torch,
)

try:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except (ImportError, OSError):
    nn = None
    DataLoader = None
    TensorDataset = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configure_cuda_runtime():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass


def _cuda_loader_kwargs(device):
    if getattr(device, "type", None) != "cuda":
        return {}

    cpu_count = os.cpu_count() or 1
    num_workers = min(8, max(2, cpu_count // 2))
    return {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


class GraphFeatureTransformer(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for GraphFeatureTransformer: {_TORCH_IMPORT_ERROR}")
        if num_features <= 0:
            raise ValueError("num_features must be greater than 0")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, num_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, max(d_model // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 1), 1),
        )

        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, X):
        X = X.unsqueeze(-1)
        X = self.value_projection(X) + self.feature_embedding
        cls = self.cls_token.expand(X.size(0), -1, -1)
        X = torch.cat([cls, X], dim=1)
        X = self.encoder(X)
        cls_output = self.norm(X[:, 0])
        return self.classifier(cls_output).squeeze(-1)


class SimpleTransformer(BaseModel):
    """
    Transformer model wrapper for numeric tabular features.

    It follows the same public API as the other model classes:
    - preprocess
    - train / train_pipeline
    - predict / predict_proba
    - evaluate / test_pipeline
    """
    def __init__(
        self,
        model_name: str = "SimpleTransformer_graph",
        device=None,
        feature_cols: Optional[List[str]] = None,
        **model_params,
    ):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for SimpleTransformer: {_TORCH_IMPORT_ERROR}")

        _configure_cuda_runtime()

        self.model_params = {
            "num_features": model_params.get("num_features"),
            "d_model": model_params.get("d_model", 64),
            "nhead": model_params.get("nhead", 4),
            "num_layers": model_params.get("num_layers", 2),
            "dim_feedforward": model_params.get("dim_feedforward", 128),
            "dropout": model_params.get("dropout", 0.15),
        }
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.scaler = RobustScaler()
        self.feature_cols = feature_cols
        self.history: list[dict] = []
        self.threshold = 0.5
        self.last_metrics: dict = {}

        transformer = None
        if self.model_params["num_features"] is not None:
            transformer = GraphFeatureTransformer(**self.model_params).to(self.device)

        super().__init__(model_name=model_name, model=transformer)

    def _init_model(self, num_features: int) -> None:
        if self.model is not None:
            return
        self.model_params["num_features"] = num_features
        self.model = GraphFeatureTransformer(**self.model_params).to(self.device)

    def _infer_feature_columns(self, data: pd.DataFrame) -> list[str]:
        drop_cols = [
            "is_reference_valid",
            "article_id",
            "ref_id",
            "vector_text_article",
            "vector_text_ref",
            "split",
        ]
        candidate = data.drop(columns=drop_cols, errors="ignore").copy()
        numeric_cols = candidate.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for SimpleTransformer")
        return numeric_cols

    def _ensure_feature_columns(self, data: pd.DataFrame, is_training: bool) -> None:
        if self.feature_cols is None:
            if not is_training:
                raise ValueError("SimpleTransformer must be fitted before preprocessing validation/test data")
            self.feature_cols = self._infer_feature_columns(data)

        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        expected_features = self.model_params.get("num_features")
        if expected_features is not None and len(self.feature_cols) != expected_features:
            raise ValueError(
                f"Expected {expected_features} feature columns, found {len(self.feature_cols)}"
            )
        self._init_model(len(self.feature_cols))

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose: bool = True) -> tuple:
        """
        Prepare numeric features as a float matrix [n_rows, n_features].
        """
        if verbose:
            print(f"[{self.model_name}] Preprocessing {len(data)} rows...")

        self._ensure_feature_columns(data, is_training=is_training)

        X = data[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = data["is_reference_valid"].to_numpy(dtype=np.float32)

        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        if verbose:
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))

        return X_scaled.astype(np.float32), y

    def _make_loader(self, X, y=None, batch_size: int = 512, shuffle: bool = False):
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        if y is not None:
            y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **_cuda_loader_kwargs(self.device),
        )

    def _compute_pos_weight(self, y):
        return compute_binary_pos_weight(torch, y, self.device)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 5,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 2,
        **kwargs,
    ):
        """
        Train the tabular-feature transformer.
        """
        if self.model is None:
            self._init_model(X_train.shape[1])

        print(f"[{self.model_name}] Starting training...")
        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._compute_pos_weight(y_train))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        best_val_f1 = -np.inf
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            row = {"epoch": epoch, "train_loss": float(total_loss / len(train_loader.dataset))}

            if X_val is not None and y_val is not None:
                val_prob = self.predict_proba(X_val, batch_size=batch_size)
                val_pred = (val_prob >= self.threshold).astype(int)
                val_f1 = f1_score(np.asarray(y_val).astype(int), val_pred, average="weighted")
                row["val_f1_weighted"] = float(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.history.append(row)
            print(row)

            if X_val is not None and y_val is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X, batch_size: int = 512):
        """
        Return positive-class probabilities.
        """
        if self.model is None:
            raise ValueError("SimpleTransformer model is not initialized.")

        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, X, batch_size: int = 512):
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)

    def evaluate(self, y_true, y_pred, title: Optional[str] = None):
        """
        Print classification metrics and plot a confusion matrix.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        metrics = evaluate_classifier_predictions(y_true, y_pred, display_labels=[0, 1], output_dict=True)
        if title:
            plt.title(title)
            plt.show()

        self.last_metrics = metrics
        return metrics

    def train_pipeline(self, raw_train, raw_val=None, **kwargs):
        """
        Complete train pipeline:
        - preprocess train graph features
        - preprocess optional validation graph features with train scaler
        - fit transformer
        - evaluate on train
        """
        verbose = kwargs.pop("verbose", True)
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, verbose=verbose)
        if raw_val is not None:
            X_val, y_val = self.preprocess(raw_val, is_training=False, verbose=False)
        else:
            X_val, y_val = None, None

        self.train(X_tr, y_tr, X_val=X_val, y_val=y_val, **kwargs)
        y_pred = self.predict(X_tr, batch_size=kwargs.get("batch_size", 512))
        return self.evaluate(y_tr, y_pred, title=f"{self.model_name} - Train confusion matrix")

    def test_pipeline(self, raw_test, batch_size: int = 512, **kwargs):
        """
        Complete test pipeline:
        - preprocess test graph features with train scaler
        - predict
        - evaluate
        """
        X_te, y_te = self.preprocess(raw_test, is_training=False, **kwargs)
        y_pred = self.predict(X_te, batch_size=batch_size)
        return self.evaluate(y_te, y_pred, title=f"{self.model_name} - Test confusion matrix")

    def save_model(
        self,
        params=None,
        df_name="graph_features",
        model_family="transformer",
        split_name="predefined_train_validation_test",
        summary=None,
        force=False,
    ):
        """
        Save a PyTorch checkpoint plus a JSON summary.
        """
        if self.model is None:
            raise ValueError("SimpleTransformer model is not initialized.")

        params_to_save = params if params is not None else self.model_params
        summary = summary if summary is not None else self.last_metrics

        df_type = df_type_from_name(df_name)
        model_slug = slug(self.model_name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        artifact_dir = PROJECT_ROOT / "Models" / df_type / model_slug
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / f"{model_slug}__{timestamp}.pt"
        summary_path = artifact_dir / f"{model_slug}__{timestamp}.json"

        if model_path.exists() and not force:
            raise FileExistsError(f"Model already exists: {model_path}")

        checkpoint = {
            "model_state_dict": self.model.cpu().state_dict(),
            "model_params": self.model_params,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "threshold": self.threshold,
            "history": self.history,
        }
        torch.save(checkpoint, model_path)
        self.model.to(self.device)

        payload = {
            "timestamp": timestamp,
            "df_type": df_type,
            "df_name": df_name,
            "model_family": model_family,
            "model_name": self.model_name,
            "split_name": split_name,
            "params": params_to_save,
            "model_path": str(model_path),
            "performance": summary,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        print("Hyperparameters:", params_to_save)
        print(f"Saved {self.model_name} model to:", model_path)
        print("Saved summary to:", summary_path)
        return model_path, summary_path


# ============================================================================
# BASELINE CLASSIFIERS
# ============================================================================

SimpleTransformerModel = SimpleTransformer

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from .base import BaseModel
from .deep_common import (
    TorchBinaryClassifierMixin,
    _TORCH_IMPORT_ERROR,
    configure_cuda_runtime,
    get_torch_device,
    loader_kwargs,
    make_dense_layers,
    nn,
    pos_weight_tensor,
    preprocess_tabular_features,
    resolve_torch_device,
    resolve_learning_rate,
    torch,
    DataLoader,
    TensorDataset,
)

try:
    import torch.nn.functional as F
except (ImportError, OSError):
    F = None


class ConvNet1D(nn.Module if nn is not None else object):
    """
    1D Convolutional Neural Network for time-series EEG data.
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        kernel_sizes: list[int] = None,
        dropout_rate: float = 0.3,
        dense_units: tuple[int, ...] = (128, 64),
        max_pool_kernel_size: int = 2,
        num_classes: int = 1,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for ConvNet1D: {_TORCH_IMPORT_ERROR}")

        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.max_pool_kernel_size = max_pool_kernel_size

        convolution_blocks = []
        for k in kernel_sizes:
            layers = [
                nn.Conv1d(input_channels, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            ]
            if max_pool_kernel_size and max_pool_kernel_size > 1:
                layers.append(nn.MaxPool1d(max_pool_kernel_size))
            convolution_blocks.append(nn.Sequential(*layers))
        self.convolutions = nn.ModuleList(convolution_blocks)

        self.fc_layers = make_dense_layers(
            input_size=num_filters * len(kernel_sizes),
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.convolutions]
        pooled = [F.adaptive_avg_pool1d(out, 1).squeeze(-1) for out in conv_outputs]
        x = torch.cat(pooled, dim=1)
        return self.fc_layers(x)


class ConvNetModel(TorchBinaryClassifierMixin, BaseModel, BaseEstimator, ClassifierMixin):
    """
    CNN model wrapper for 1D EEG time-series data.
    """
    def __init__(
        self,
        model_name: str = "CNN1D",
        device=None,
        input_channels: int = 1,
        num_filters: int = 32,
        kernel_sizes: list[int] = None,
        dropout_rate: float = 0.3,
        dense_units: tuple[int, ...] = (128, 64),
        max_pool_kernel_size: int = 2,
        num_classes: int = 1,
        scaler=None,
        feature_cols=None,
        num_workers: int = 0,
        pin_memory: bool = None,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        **kwargs
    ):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for ConvNetModel: {_TORCH_IMPORT_ERROR}")

        configure_cuda_runtime()

        self.device = resolve_torch_device(device)
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else [3, 5, 7]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.max_pool_kernel_size = max_pool_kernel_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        cnn = ConvNet1D(
            input_channels=input_channels,
            num_filters=num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            max_pool_kernel_size=max_pool_kernel_size,
            num_classes=num_classes,
        ).to(self.device)

        super().__init__(model_name=model_name, model=cnn)

        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.feature_cols = feature_cols
        self.history = []
        self.threshold = 0.5

    def preprocess(self, data, is_training: bool = True, verbose=True) -> tuple:
        return preprocess_tabular_features(self, data, is_training, verbose)

    def _make_loader(self, X, y=None, batch_size: int = 128, shuffle: bool = False):
        X_reshaped = torch.from_numpy(X).unsqueeze(1)
        if y is not None:
            y_tensor = torch.from_numpy(y.astype(np.float32))
            dataset = TensorDataset(X_reshaped, y_tensor)
        else:
            dataset = TensorDataset(X_reshaped)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **loader_kwargs(
                self.device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            ),
        )

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        patience: int = 3,
        show_learning_curve: bool = True,
        **kwargs
    ):
        print(f"[{self.model_name}] Starting training...")
        learning_rate = resolve_learning_rate(learning_rate, kwargs)

        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor(
                y_train,
                self.device,
                class_weights=kwargs.pop("class_weights", None),
                pos_weight=kwargs.pop("pos_weight", None),
            )
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
                    loss = criterion(logits.view(-1), y_batch.view(-1))

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
                val_loader = self._make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        y_batch = y_batch.to(self.device, non_blocking=True)
                        with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                            logits = self.model(X_batch)
                            loss = criterion(logits.view(-1), y_batch.view(-1))
                        val_loss += loss.item() * X_batch.size(0)

                val_pred = (self.predict_proba(X_val, batch_size=batch_size) >= self.threshold).astype(int)
                val_f1 = f1_score(y_val.astype(int), val_pred, average="weighted")
                row["val_loss"] = float(val_loss / len(val_loader.dataset))
                row["val_f1"] = float(val_f1)

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
        self._plot_learning_curve_after_training(show_learning_curve)

    def predict_proba(self, X, batch_size: int = 128):
        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []

        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        return np.concatenate(probabilities).squeeze()

    def predict(self, X, batch_size: int = 128):
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        self.train(X, y, show_learning_curve=False)
        return self

    def score(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        pred = self.predict(X)
        return f1_score(y, pred, average="weighted", zero_division=0)

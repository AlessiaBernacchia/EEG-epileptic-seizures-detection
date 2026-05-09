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


class RecurrentNet(nn.Module if nn is not None else object):
    """
    Recurrent Neural Network (LSTM) for time-series EEG data.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        dense_units: tuple[int, ...] = (128, 64),
        num_classes: int = 1,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for RecurrentNet: {_TORCH_IMPORT_ERROR}")

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dense_units = dense_units

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc_layers = make_dense_layers(
            input_size=lstm_output_size,
            dense_units=dense_units,
            dropout_rate=dropout,
            num_classes=num_classes,
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc_layers(last_out)


class RecurrentNetModel(TorchBinaryClassifierMixin, BaseModel, BaseEstimator, ClassifierMixin):
    """
    LSTM model wrapper for time-series EEG data.
    """
    def __init__(
        self,
        model_name: str = "LSTM",
        device=None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        dense_units: tuple[int, ...] = (128, 64),
        sequence_length: int = 1,
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
            raise OSError(f"PyTorch is required for RecurrentNetModel: {_TORCH_IMPORT_ERROR}")

        configure_cuda_runtime()

        self.device = resolve_torch_device(device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.dense_units = dense_units
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.input_size = None
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        super().__init__(model_name=model_name, model=None)

        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.feature_cols = feature_cols
        self.history = []
        self.threshold = 0.5

    def _init_model(self, input_size: int):
        if self.model is not None:
            return

        self.input_size = input_size
        self.model = RecurrentNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            dense_units=self.dense_units,
            num_classes=self.num_classes,
        ).to(self.device)

    def preprocess(self, data, is_training: bool = True, verbose=True) -> tuple:
        return preprocess_tabular_features(self, data, is_training, verbose)

    def _reshape_sequence(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            return X

        if self.sequence_length <= 1:
            return X[:, None, :]

        if X.shape[1] % self.sequence_length != 0:
            raise ValueError(
                f"{self.model_name}: cannot reshape {X.shape[1]} features into "
                f"sequence_length={self.sequence_length}. The feature count must be divisible."
            )

        input_size = X.shape[1] // self.sequence_length
        return X.reshape(X.shape[0], self.sequence_length, input_size)

    def _make_loader(self, X, y=None, batch_size: int = 128, shuffle: bool = False):
        X_reshaped = torch.from_numpy(self._reshape_sequence(X))

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
        if self.model is None:
            self._init_model(self._reshape_sequence(X_train).shape[2])

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
        if self.model is None:
            raise ValueError("LSTM model not initialized. Call train() first.")

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

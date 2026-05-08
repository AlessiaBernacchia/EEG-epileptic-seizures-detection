import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .base import BaseModel, _TORCH_IMPORT_ERROR, get_torch_device, torch

try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except (ImportError, OSError):
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


class ConvNet1D(nn.Module if nn is not None else object):
    """
    1D Convolutional Neural Network for time-series EEG data.
    Designed to extract temporal patterns from raw EEG signals or features.
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        kernel_sizes: list[int] = None,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for ConvNet1D: {_TORCH_IMPORT_ERROR}")
        
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        
        # Multiple parallel convolutions with different kernel sizes
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, num_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            for k in kernel_sizes
        ])
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Apply parallel convolutions
        conv_outputs = [conv(x) for conv in self.convolutions]
        # Global average pooling
        pooled = [F.adaptive_avg_pool1d(out, 1).squeeze(-1) for out in conv_outputs]
        # Concatenate
        x = torch.cat(pooled, dim=1)
        # Fully connected layers
        return self.fc_layers(x)
class ConvNetModel(BaseModel):
    """
    CNN model wrapper for 1D EEG time-series data.
    Follows the same API as other models.
    """
    def __init__(
        self,
        model_name: str = "CNN1D",
        device=None,
        input_channels: int = 1,
        num_filters: int = 32,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for ConvNetModel: {_TORCH_IMPORT_ERROR}")
        
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        
        cnn = ConvNet1D(
            input_channels=input_channels,
            num_filters=num_filters,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        super().__init__(model_name=model_name, model=cnn)
        
        self.scaler = None  # Scaled in normalization step
        self.history = []
        self.threshold = 0.5

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Preprocess data for CNN: reshape features as time-series.
        Assumes data has shape (n_samples, n_features).
        Reshapes to (n_samples, 1, n_features) for 1D convolution.
        """
        feature_cols = [col for col in data.columns if col not in ['label', 'target', 'is_seizure']]
        if not feature_cols:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for col in ['label', 'target', 'is_seizure']:
                if col in feature_cols:
                    feature_cols.remove(col)
        
        X = data[feature_cols].fillna(0).to_numpy().astype(np.float32)
        
        # Normalize to [0, 1]
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Get labels
        for label_col in ['is_seizure', 'label', 'target', 'y']:
            if label_col in data.columns:
                y = data[label_col].to_numpy()
                break
        else:
            y = data.iloc[:, -1].to_numpy()
        
        if verbose:
            print(f"[{self.model_name}] Features shape: {X.shape}, Labels shape: {y.shape}")
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))
        
        return X.astype(np.float32), y

    def _make_loader(self, X, y=None, batch_size: int = 32, shuffle: bool = False):
        # Reshape for CNN: (n_samples, n_features) -> (n_samples, 1, n_features)
        X_reshaped = torch.from_numpy(X).unsqueeze(1)  # Add channel dimension
        if y is not None:
            y_tensor = torch.from_numpy(y.astype(np.float32))
            dataset = TensorDataset(X_reshaped, y_tensor)
        else:
            dataset = TensorDataset(X_reshaped)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda"
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 10,
              batch_size: int = 32, learning_rate: float = 1e-3, patience: int = 3, **kwargs):
        """Train the CNN model."""
        print(f"[{self.model_name}] Starting training...")
        
        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
                
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
            
            row = {"epoch": epoch, "train_loss": float(total_loss / len(train_loader.dataset))}
            
            if X_val is not None and y_val is not None:
                val_pred = (self.predict_proba(X_val, batch_size=batch_size) >= self.threshold).astype(int)
                val_f1 = f1_score(y_val.astype(int), val_pred, average="weighted")
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

    def predict_proba(self, X, batch_size: int = 32):
        """Return probability scores."""
        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []
        
        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(probabilities).squeeze()

    def predict(self, X, batch_size: int = 32):
        """Predict class labels."""
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)
class RecurrentNet(nn.Module if nn is not None else object):
    """
    Recurrent Neural Network (LSTM) for time-series EEG data.
    Designed to capture temporal dependencies in EEG signals.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for RecurrentNet: {_TORCH_IMPORT_ERROR}")
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # LSTM output: (batch_size, seq_length, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_out = lstm_out[:, -1, :]
        # Fully connected
        return self.fc_layers(last_out)
class RecurrentNetModel(BaseModel):
    """
    LSTM model wrapper for time-series EEG data.
    Follows the same API as other models.
    """
    def __init__(
        self,
        model_name: str = "LSTM",
        device=None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        **kwargs
    ):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for RecurrentNetModel: {_TORCH_IMPORT_ERROR}")
        
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.input_size = None
        
        # Initialize with dummy model
        super().__init__(model_name=model_name, model=None)
        
        self.scaler = None
        self.history = []
        self.threshold = 0.5

    def _init_model(self, input_size: int):
        """Initialize LSTM model with correct input size."""
        if self.model is not None:
            return
        
        self.input_size = input_size
        rnn = RecurrentNet(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        self.model = rnn

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Preprocess data for RNN: reshape features as sequence.
        Assumes data has shape (n_samples, n_features).
        Reshapes to (n_samples, 1, n_features) for sequence input.
        """
        feature_cols = [col for col in data.columns if col not in ['label', 'target', 'is_seizure']]
        if not feature_cols:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for col in ['label', 'target', 'is_seizure']:
                if col in feature_cols:
                    feature_cols.remove(col)
        
        X = data[feature_cols].fillna(0).to_numpy().astype(np.float32)
        
        # Normalize to [0, 1]
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X = (X - X_min) / (X_max - X_min + 1e-8)
        
        # Get labels
        for label_col in ['is_seizure', 'label', 'target', 'y']:
            if label_col in data.columns:
                y = data[label_col].to_numpy()
                break
        else:
            y = data.iloc[:, -1].to_numpy()
        
        if verbose:
            print(f"[{self.model_name}] Features shape: {X.shape}, Labels shape: {y.shape}")
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))
        
        return X.astype(np.float32), y

    def _make_loader(self, X, y=None, batch_size: int = 32, shuffle: bool = False):
        # Reshape for RNN: (n_samples, n_features) -> (n_samples, 1, n_features)
        # where 1 is the sequence length
        X_reshaped = torch.from_numpy(X).unsqueeze(1)  # Add sequence dimension
        
        if y is not None:
            y_tensor = torch.from_numpy(y.astype(np.float32))
            dataset = TensorDataset(X_reshaped, y_tensor)
        else:
            dataset = TensorDataset(X_reshaped)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda"
        )

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 10,
              batch_size: int = 32, learning_rate: float = 1e-3, patience: int = 3, **kwargs):
        """Train the LSTM model."""
        if self.model is None:
            self._init_model(X_train.shape[1])
        
        print(f"[{self.model_name}] Starting training...")
        
        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
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
                
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
            
            row = {"epoch": epoch, "train_loss": float(total_loss / len(train_loader.dataset))}
            
            if X_val is not None and y_val is not None:
                val_pred = (self.predict_proba(X_val, batch_size=batch_size) >= self.threshold).astype(int)
                val_f1 = f1_score(y_val.astype(int), val_pred, average="weighted")
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

    def predict_proba(self, X, batch_size: int = 32):
        """Return probability scores."""
        if self.model is None:
            raise ValueError("LSTM model not initialized. Call train() first.")
        
        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []
        
        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(probabilities).squeeze()

    def predict(self, X, batch_size: int = 32):
        """Predict class labels."""
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)

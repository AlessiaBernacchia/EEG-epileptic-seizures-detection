from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import re

RANDOM_STATE = 42
DEVICE = "cpu"
N_JOBS = -1


def _is_cuda_device(device) -> bool:
    return str(device).lower().startswith("cuda")


def prepare_scaled_tabular_features(
    data: pd.DataFrame,
    scaler,
    is_training: bool = True,
    as_dataframe: bool = False,
    verbose: bool = True,
    model_name: str = "model",
):
    label_candidates = ["is_seizure"]
    label_col = next((col for col in label_candidates if col in data.columns), data.columns[-1])

    drop_cols = {
        label_col,
    }

    feature_cols = [col for col in data.columns if col not in drop_cols]
    X = data[feature_cols].select_dtypes(include=[np.number, "bool"]).copy()

    if X.empty:
        raise ValueError(f"[{model_name}] No numeric feature columns found")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = data[label_col]

    if scaler is not None:
        if is_training:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
    else:
        X_scaled = X.to_numpy()

    if as_dataframe:
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=data.index)

    if verbose:
        print(f"[{model_name}] Features shape: {X_scaled.shape}, Labels shape: {y.shape}")
        print("Label distribution:")
        print(pd.Series(y).value_counts(normalize=True))

    return X_scaled, y


def evaluate_classifier_predictions(y_true, y_pred, display_labels=None, output_dict: bool = False):
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
    import matplotlib.pyplot as plt

    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=display_labels) if display_labels is not None else confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot()
    plt.show()

    if output_dict:
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return None


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def df_type_from_name(df_name: str) -> str:
    return str(df_name).replace(" ", "_").lower()


class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models in the pipeline.
    Ensures that every model implements mandatory preprocessing and prediction logic.
    """
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self.model = model

    @abstractmethod
    def preprocess(self, data: pd.DataFrame, is_training=True) -> tuple:
        """
        Cleaning, feature engineering, and/or scaling required for the model.
        :param data: The raw input dataframe.
        :param is_training: Boolean to distinguish between training and inference (e.g., for fit_transform vs transform).
        :return: Tuple (X, y) ready for the model.
        """
        pass

    def train(self, X_train, y_train, **kwargs):
        """
        Fits the model to the training data.
        """
        if self.model is None:
            raise ValueError(f"Model for {self.model_name} is not initialized.")
        print(f"[{self.model_name}] Starting training...")
        self.model.fit(X_train, y_train, **kwargs)

    def grid_search(self):
        raise Exception('NO GRID SEARCH DEFINED!!')

    def hypertune_pipeline(self, df_train, df_val, param_grid, n_jobs=N_JOBS, frac=1, **kwargs):
        """
        Hypertune, find the best parameters.
        :param df_train: Dataframe with train features and targets.
        :param df_val: Dtaaframe with validation features and targets.
        :param param_grid: params to tune.
        """
        # Guard against accidental forwarding of training-only args to GridSearchCV.
        frac = kwargs.pop("frac", frac)
        grid_search = self.grid_search(df_train, df_val, param_grid, n_jobs=n_jobs, **kwargs)
        # final training creating model with best params
        best_params = grid_search.best_params_
        self.model.set_params(**best_params)
        print(f'[{self.model_name}] Train model with best params...')
        self.train_pipeline(df_train, frac=frac)
        return grid_search

    def predict(self, X):
        """
        Inference: Uses the trained model to make predictions.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Calculates and prints performance metrics.
        """
        display_labels = getattr(self.model, "classes_", None)
        evaluate_classifier_predictions(y_true, y_pred, display_labels=display_labels)


    def train_pipeline(self, raw_train, frac=0.01, random_state=RANDOM_STATE, **kwargs):
        """
        Complete pipeline for train:
        - preprocess for training
        - fitting the model
        - prediction
        - evaluation
        """
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, **kwargs)
        self.train(X_tr, y_tr)

        if frac < 1:
            new_size = int(len(X_tr) * frac)
            print(f"Selected {new_size}/{len(X_tr)}")
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_tr), size=new_size, replace=False)

            X_sample = X_tr[idx]
            y_sample = y_tr.iloc[idx] if hasattr(y_tr, "iloc") else y_tr[idx]

            X_tr = X_sample
            y_tr = y_sample
        elif frac > 1:
            raise ValueError("Frac value not Valid (should be < 1)")


        y_pred = self.predict(X_tr)
        self.evaluate(y_tr, y_pred)

    def test_pipeline(self, raw_test, **kwargs):
        """
        Complete pipeline for test:
        - preprocess for training
        - prediction
        - evaluation
        """
        X_te, y_te = self.preprocess(raw_test, is_training=False, **kwargs)

        y_pred = self.predict(X_te)
        self.evaluate(y_te, y_pred)

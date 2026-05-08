import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb

from .base import (
    BaseModel,
    DEVICE,
    N_JOBS,
    RANDOM_STATE,
    _is_cuda_device,
    prepare_scaled_tabular_features,
)


class KNNModel(BaseModel):
    """
    Implementation of the KNN Baseline model using paper features.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="KNN", n_jobs=N_JOBS, **kwargs):
        # Initialize the scikit-learn KNN model
        knn_internal = KNeighborsClassifier(
            n_jobs=n_jobs,
            **kwargs
        )
        super().__init__(model_name=model_name, model=knn_internal)

        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, max_tuning_samples=50000, n_jobs=N_JOBS, **kwargs) -> list:
        """
        Hypertune, find the best parameters.
        :param X_val: Dataframe with validation features.
        :param y_val: Dtaaframe with validation targets.
        :param param_grid: params to tune.
        """
        print(f'[{self.model_name}] Grid Search...')
        # Downsampling for speed
        def downsample_indices(indices, n_samples):
            if len(indices) > n_samples:
                return np.random.choice(indices, n_samples, replace=False)
            return indices

        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        # sample for tuning
        train_tuning_idx = downsample_indices(np.arange(len(X_train_scaled)), int(max_tuning_samples * 0.8))
        val_tuning_idx = downsample_indices(np.arange(len(X_val_scaled)), int(max_tuning_samples * 0.2))

        X_subset = np.vstack((X_train_scaled[train_tuning_idx], X_val_scaled[val_tuning_idx]))
        y_subset = np.concatenate((y_train.iloc[train_tuning_idx], y_val.iloc[val_tuning_idx]))

        # Create split indices: -1 for train, 0 for validation
        split_index = np.concatenate([-1 * np.ones(len(train_tuning_idx)), 0 * np.ones(len(val_tuning_idx))])
        ps = PredefinedSplit(test_fold=split_index)

        # GridSearchCV
        print(f"\nStarting tuning on {len(X_subset)} samples...")
        grid_search = GridSearchCV(
            KNeighborsClassifier(n_jobs=n_jobs),
            param_grid=param_grid,
            cv=ps,
            n_jobs=n_jobs,
            **kwargs
        )
        grid_search.fit(X_subset, y_subset)

        # print results
        best_params = grid_search.best_params_
        print("\nBest parameters found:")
        print(best_params)

        # Final training on the full dataset with the best parameters
        model = grid_search.best_estimator_
        model.fit(X_train_scaled, y_train)

        print(f"\nOptimal model ready: {model}")

        return grid_search

    def predict_proba(self, X):
        """
        Get probability scores (useful for AUC calculation).
        """
        return self.model.predict_proba(X)


class XGBModel(BaseModel):
    """
    Implementation of the XGB Baseline model using paper features.
    Inherits from BaseModel.
    """
    def __init__(self, model_name='XGB', device=DEVICE, **kwargs):
        # Initialize the scikit-learn KNN model
        xgb_internal = XGBClassifier(
            device=device,
            **kwargs
        )
        super().__init__(model_name=model_name, model=xgb_internal)

        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, device=DEVICE, n_jobs=N_JOBS, **kwargs):
        """
        Hypertune, find the best parameters.
        :param X_val: Dataframe with validation features.
        :param y_val: Dtaaframe with validation targets.
        :param param_grid: params to tune.
        """

        print(f'[{self.model_name}] Grid Search...')
        # preprocess data
        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        search_n_jobs = 1 if _is_cuda_device(device) else n_jobs
        if _is_cuda_device(device) and n_jobs != 1:
            print(
                f"[{self.model_name}] CUDA detected: using n_jobs=1 for RandomizedSearchCV "
                "to avoid running multiple GPU fits at the same time."
            )

        model_params = self.model.get_params()
        model_params.update({
            "tree_method": model_params.get("tree_method") or "hist",
            "device": device,
        })
        if _is_cuda_device(device):
            model_params["n_jobs"] = 1

        model = XGBClassifier(**model_params)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_jobs=search_n_jobs,
            **kwargs
        )

        # Run randomized search on the validation split (kept small by earlier sampling)
        random_search.fit(X_val_scaled, y_val)

        best_params = random_search.best_params_
        print("\nBest parameters found:")
        print(best_params)

        print(f"\nOptimal model ready: {random_search.best_estimator_}")

        return random_search


    def predict_proba(self, X):
        """
        Get probability scores (useful for AUC calculation).
        """
        return self.model.predict_proba(X)


class LGBModel(BaseModel):
    """
    Implementation of the LightGBM model for large-scale embedding classification.
    Optimized for high speed and parallelization.
    """
    def __init__(self, model_name='LGBM', device=DEVICE, n_jobs=N_JOBS, random_state=RANDOM_STATE, **kwargs):
        # Initialize LightGBM Classifier
        # device can be 'cpu' or 'gpu'
        lgb_internal = lgb.LGBMClassifier(
            device=device,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        super().__init__(model_name=model_name, model=lgb_internal)
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            as_dataframe=True,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, n_iter=15, n_jobs=N_JOBS, **kwargs):
        """
        Hyperparameter tuning using RandomizedSearchCV for efficiency.
        """
        print(f'[{self.model_name}] Starting Randomized Search...')
        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        # We use a subset for tuning to speed up the process,
        # but LGBM is fast enough to handle larger chunks than KNN
        model = lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            **kwargs
        )

        # Tuning on validation to find best params quickly
        random_search.fit(X_val_scaled, y_val)

        print(f"\nBest parameters: {random_search.best_params_}")

        # Final training on the full training set
        self.model = random_search.best_estimator_
        self.model.fit(X_train_scaled, y_train)

        return random_search

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="LogisticRegression", n_jobs=N_JOBS, **kwargs):
        lr_model = LogisticRegression(
            n_jobs=n_jobs,
            max_iter=1000,
            random_state=RANDOM_STATE,
            **kwargs
        )
        super().__init__(model_name=model_name, model=lr_model)
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]


class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="NaiveBayes", **kwargs):
        nb_model = GaussianNB(**kwargs)
        super().__init__(model_name=model_name, model=nb_model)
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]


class SVMModel(BaseModel):
    """
    Support Vector Machine baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="SVM", probability=True, **kwargs):
        svm_model = SVC(
            probability=probability,
            random_state=RANDOM_STATE,
            **kwargs
        )
        super().__init__(model_name=model_name, model=svm_model)
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]


class RandomForestModel(BaseModel):
    """
    Random Forest baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="RandomForest", n_jobs=N_JOBS, **kwargs):
        rf_model = RandomForestClassifier(
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            **kwargs
        )
        super().__init__(model_name=model_name, model=rf_model)
        self.scaler = None  # RF doesn't require scaling

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features without scaling."""
        # Extract feature columns and labels
        feature_cols = [col for col in data.columns if col not in ['label', 'target', 'is_seizure']]
        if not feature_cols:
            # Try to infer from data structure
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for col in ['label', 'target', 'is_seizure']:
                if col in feature_cols:
                    feature_cols.remove(col)

        X = data[feature_cols].fillna(0).to_numpy()
        # Try multiple possible label columns
        for label_col in ['is_seizure', 'label', 'target', 'y']:
            if label_col in data.columns:
                y = data[label_col].to_numpy()
                break
        else:
            # Assume last column is label
            y = data.iloc[:, -1].to_numpy()

        if verbose:
            print(f"[{self.model_name}] Features shape: {X.shape}, Labels shape: {y.shape}")
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))

        return X, y

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]
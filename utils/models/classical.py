import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from .base import (
    BaseModel,
    DEVICE,
    N_JOBS,
    RANDOM_STATE,
    _is_cuda_device,
    prepare_scaled_tabular_features,
)


def _stack_train_val(X_train, y_train, X_val, y_val):
    if isinstance(X_train, pd.DataFrame):
        X = pd.concat([X_train, X_val], axis=0)
    else:
        X = np.vstack((X_train, X_val))
    y = np.concatenate((np.asarray(y_train), np.asarray(y_val)))
    split_index = np.concatenate((
        -1 * np.ones(len(y_train), dtype=int),
        np.zeros(len(y_val), dtype=int),
    ))
    return X, y, PredefinedSplit(test_fold=split_index)


def _search_on_train_val(
    estimator,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    n_iter=None,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
    search_type="grid",
    **kwargs,
):
    X, y, split = _stack_train_val(X_train, y_train, X_val, y_val)
    scoring = kwargs.pop("scoring", "f1_weighted")

    if search_type == "random" and n_iter is not None:
        search = RandomizedSearchCV(
            clone(estimator),
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=split,
            n_jobs=n_jobs,
            random_state=random_state,
            scoring=scoring,
            **kwargs,
        )
    else:
        search = GridSearchCV(
            clone(estimator),
            param_grid=param_grid,
            cv=split,
            n_jobs=n_jobs,
            scoring=scoring,
            **kwargs,
        )

    search.fit(X, y)
    print(f"\nBest parameters found for {estimator.__class__.__name__}:")
    print(search.best_params_)
    return search


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
        
    '''
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
    '''

    def grid_search(
        self,
        df_train,
        df_val,
        param_grid,
        n_iter=None,
        max_tuning_samples=50000,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        **kwargs,
    ):
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
                rng = np.random.default_rng(random_state)
                return rng.choice(indices, n_samples, replace=False)
            return indices

        # sample for tuning
        train_tuning_idx = downsample_indices(np.arange(len(df_train)), int(max_tuning_samples * 0.8))
        val_tuning_idx = downsample_indices(np.arange(len(df_val)), int(max_tuning_samples * 0.2))

        print(f"\nStarting tuning on {len(train_tuning_idx) + len(val_tuning_idx)} samples...")
        grid_search = _search_on_train_val(
            self.model,
            df_train.iloc[train_tuning_idx] if hasattr(df_train, 'iloc') else df_train[train_tuning_idx],
            df_train.iloc[train_tuning_idx]['is_seizure'] if hasattr(df_train, 'iloc') else df_train[train_tuning_idx]['is_seizure'],
            df_val.iloc[val_tuning_idx] if hasattr(df_val, 'iloc') else df_val[val_tuning_idx],
            df_val.iloc[val_tuning_idx]['is_seizure'] if hasattr(df_val, 'iloc') else df_val[val_tuning_idx]['is_seizure'],
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_type="random" if n_iter else "grid",
            random_state=random_state,
            **kwargs,
        )
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
        if XGBClassifier is None:
            raise ImportError("XGBModel requires xgboost. Install it or skip this model.")
        # Initialize the scikit-learn KNN model
        xgb_internal = XGBClassifier(
            device=device,
            **kwargs
        )
        super().__init__(model_name=model_name, model=xgb_internal)

        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = RobustScaler()
    
    '''
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
    '''
    
    def grid_search(
        self,
        df_train,
        df_val,
        param_grid,
        device=DEVICE,
        n_iter=15,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        **kwargs,
    ):
        """
        Hypertune, find the best parameters.
        :param X_val: Dataframe with validation features.
        :param y_val: Dtaaframe with validation targets.
        :param param_grid: params to tune.
        """

        print(f'[{self.model_name}] Grid Search...')

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
        
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]

        random_search = _search_on_train_val(
            model,
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=search_n_jobs,
            random_state=random_state,
            search_type="random",
            **kwargs,
        )
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
        if lgb is None:
            raise ImportError("LGBModel requires lightgbm. Install it or skip this model.")
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

    
    '''
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
    '''

    def grid_search(
        self,
        df_train,
        df_val,
        param_grid,
        n_iter=12,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        scoring="f1_weighted",
        **kwargs,
    ):
        """
        Hypertune LightGBM on the fixed train/validation split.
        """

        print(f"[{self.model_name}] Starting Randomized Search...")

        model_params = self.model.get_params()
        if n_jobs != 1:
            model_params["n_jobs"] = 1
            
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]

        search = _search_on_train_val(
            lgb.LGBMClassifier(**model_params),
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            search_type="random",
            scoring=scoring,
            **kwargs,
        )

        self.model = search.best_estimator_

        print(f"[{self.model_name}] Best params: {search.best_params_}")
        print(f"[{self.model_name}] Best validation score: {search.best_score_:.4f}")

        return search
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="LogisticRegression", n_jobs=N_JOBS, **kwargs):
        params = {"n_jobs": n_jobs, "max_iter": 1000, "random_state": RANDOM_STATE}
        params.update(kwargs)
        lr_model = LogisticRegression(**params)
        super().__init__(model_name=model_name, model=lr_model)
        self.scaler = RobustScaler()

    '''
    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )
    '''

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]

    def grid_search(self, df_train, df_val, param_grid, n_iter=None, n_jobs=N_JOBS, **kwargs):
        print(f'[{self.model_name}] Grid Search...')
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]
        return _search_on_train_val(
            self.model,
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_type="random" if n_iter else "grid",
            **kwargs,
        )


class NaiveBayesModel(BaseModel):
    """
    Gaussian Naive Bayes baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="NaiveBayes", **kwargs):
        nb_model = GaussianNB(**kwargs)
        super().__init__(model_name=model_name, model=nb_model)
        self.scaler = RobustScaler()

    '''
    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )
    '''

    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]

    def grid_search(self, df_train, df_val, param_grid, n_iter=None, n_jobs=N_JOBS, **kwargs):
        print(f'[{self.model_name}] Grid Search...')
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]
        return _search_on_train_val(
            self.model,
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_type="random" if n_iter else "grid",
            **kwargs,
        )


class SVMModel(BaseModel):
    """
    Support Vector Machine baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="SVM", probability=True, **kwargs):
        params = {"probability": probability, "random_state": RANDOM_STATE}
        params.update(kwargs)
        svm_model = SVC(**params)
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

    def grid_search(self, df_train, df_val, param_grid, n_iter=None, n_jobs=N_JOBS, **kwargs):
        print(f'[{self.model_name}] Grid Search...')
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]
        return _search_on_train_val(
            self.model,
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_type="random" if n_iter else "grid",
            **kwargs,
        )


class RandomForestModel(BaseModel):
    """
    Random Forest baseline model for EEG seizure detection.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="RandomForest", n_jobs=N_JOBS, **kwargs):
        params = {"n_jobs": n_jobs, "random_state": RANDOM_STATE}
        params.update(kwargs)
        rf_model = RandomForestClassifier(**params)
        super().__init__(model_name=model_name, model=rf_model)
        self.scaler = None  # RF doesn't require scaling

    '''
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
    '''
    
    def predict_proba(self, X):
        """Get probability scores."""
        return self.model.predict_proba(X)[:, 1]

    def grid_search(self, df_train, df_val, param_grid, n_iter=None, n_jobs=N_JOBS, **kwargs):
        print(f'[{self.model_name}] Grid Search...')
        y_train = df_train.is_seizure if 'is_seizure' in df_train.columns else df_train.iloc[:, -1]
        y_val = df_val.is_seizure if 'is_seizure' in df_val.columns else df_val.iloc[:, -1]
        return _search_on_train_val(
            self.model,
            df_train,
            y_train,
            df_val,
            y_val,
            param_grid,
            n_iter=n_iter,
            n_jobs=n_jobs,
            search_type="random" if n_iter else "grid",
            **kwargs,
        )

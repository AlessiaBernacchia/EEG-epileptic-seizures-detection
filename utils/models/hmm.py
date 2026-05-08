import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from .base import BaseModel, RANDOM_STATE, prepare_scaled_tabular_features


class HiddenMarkovModel(BaseModel):
    """
    Hidden Markov Model for EEG seizure detection.
    Uses hmmlearn library to model sequential patterns in EEG data.
    """
    def __init__(
        self,
        model_name="HMM",
        n_components=3,
        covariance_type="full",
        n_iter=100,
        **kwargs
    ):
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HiddenMarkovModel. Install with: pip install hmmlearn")
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.hmm_seizure = None
        self.hmm_normal = None
        self.scaler = RobustScaler()
        
        # Create a dummy model for the interface
        super().__init__(model_name=model_name, model=None)

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """Prepares features with scaling."""
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def train(self, X_train, y_train, **kwargs):
        """
        Train separate HMM models for seizure and normal data.
        """
        print(f"[{self.model_name}] Starting training...")
        
        seizure_data = X_train[y_train == 1]
        normal_data = X_train[y_train == 0]
        
        print(f"  Seizure samples: {len(seizure_data)}, Normal samples: {len(normal_data)}")
        
        # Train seizure model
        if len(seizure_data) > 0:
            self.hmm_seizure = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=RANDOM_STATE
            )
            self.hmm_seizure.fit(seizure_data)
            print(f"  Seizure HMM trained with log-likelihood: {self.hmm_seizure.score(seizure_data):.4f}")
        
        # Train normal model
        if len(normal_data) > 0:
            self.hmm_normal = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=RANDOM_STATE
            )
            self.hmm_normal.fit(normal_data)
            print(f"  Normal HMM trained with log-likelihood: {self.hmm_normal.score(normal_data):.4f}")

    def predict_proba(self, X):
        """
        Predict probability of seizure using log-likelihood ratio.
        """
        if self.hmm_seizure is None or self.hmm_normal is None:
            raise ValueError("HMM models not trained. Call train() first.")
        
        seizure_scores = np.array([self.hmm_seizure.score(x.reshape(1, -1)) for x in X])
        normal_scores = np.array([self.hmm_normal.score(x.reshape(1, -1)) for x in X])
        
        # Compute probability using softmax on scores
        log_likelihood_ratio = seizure_scores - normal_scores
        proba = 1.0 / (1.0 + np.exp(-log_likelihood_ratio))
        
        return proba

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

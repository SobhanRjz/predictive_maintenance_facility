"""XGBoost model implementation."""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from xgboost import XGBClassifier
from src.core.interfaces import IModel


class XGBoostModel(IModel):
    """XGBoost classifier for health status prediction."""
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        target_col: str = 'health_status',
        random_state: int = 42,
        early_stopping_rounds: int = None,
        eval_metric: str = 'mlogloss'
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            target_col: Name of target column
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds (None to disable)
            eval_metric: Evaluation metric for early stopping
        """
        self._target_col = target_col
        self._early_stopping_rounds = early_stopping_rounds
        self._eval_metric = eval_metric

        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric=eval_metric,
            scale_pos_weight=[1, 28, 62]
        )
        self._label_encoder = LabelEncoder()
        self._is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train XGBoost model with class weights."""
        y_encoded = self._label_encoder.fit_transform(y_train)

        # Calculate class weights to handle imbalance
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)
        
        print(f"Class distribution in training: {pd.Series(y_train).value_counts().to_dict()}")

        if X_val is not None and y_val is not None and self._early_stopping_rounds is not None:
            # Use early stopping with validation data
            y_val_encoded = self._label_encoder.transform(y_val)
            self._model.fit(
                X_train, y_encoded,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val_encoded)],
                early_stopping_rounds=self._early_stopping_rounds,
                verbose=True
            )
            print(f"Training stopped early at {self._model.best_iteration} rounds")
        else:
            # Train without early stopping with class weights
            self._model.fit(X_train, y_encoded, sample_weight=sample_weights, verbose=True)

        self._is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        y_pred_encoded = self._model.predict(X)
        return self._label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        return self._model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def save(self, path: str) -> None:
        """Save model and label encoder to disk."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self._model,
            'label_encoder': self._label_encoder,
            'is_fitted': self._is_fitted
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model and label encoder from disk."""
        model_data = joblib.load(path)
        self._model = model_data['model']
        self._label_encoder = model_data['label_encoder']
        self._is_fitted = model_data['is_fitted']


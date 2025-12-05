"""Base class for future Deep Learning models."""
import numpy as np
from src.core.interfaces import IModel


class BaseDLModel(IModel):
    """
    Base class for DL models (LSTM, CNN, Transformer, etc.).
    
    Future implementations should inherit from this class.
    """
    
    def __init__(self, target_col: str = 'health_status'):
        self._target_col = target_col
        self._model = None
        self._is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train DL model."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        raise NotImplementedError("Subclasses must implement load()")


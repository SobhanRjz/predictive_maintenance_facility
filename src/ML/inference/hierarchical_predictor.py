"""Hierarchical inference pipeline for multi-layer classification."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.ML.models.xgboost_model import XGBoostModel


class HierarchicalPredictor:
    """Executes hierarchical prediction: Layer1 -> Layer2 (conditional)."""
    
    def __init__(
        self,
        layer1_model_path: str,
        layer2_warning_model_path: str = None,
        layer2_failure_model_path: str = None
    ):
        """
        Args:
            layer1_model_path: Path to layer1 anomaly detection model
            layer2_warning_model_path: Path to warning type classifier (optional)
            layer2_failure_model_path: Path to failure type classifier (optional)
        """
        self._layer1 = XGBoostModel()
        self._layer1.load(layer1_model_path)
        
        self._layer2_warning = None
        if layer2_warning_model_path and Path(layer2_warning_model_path).exists():
            self._layer2_warning = XGBoostModel()
            self._layer2_warning.load(layer2_warning_model_path)
        
        self._layer2_failure = None
        if layer2_failure_model_path and Path(layer2_failure_model_path).exists():
            self._layer2_failure = XGBoostModel()
            self._layer2_failure.load(layer2_failure_model_path)
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Hierarchical prediction with conditional layer2 classification.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            DataFrame with predictions: health_status, warning_type, failure_type
        """
        # Prepare features for layer1
        X_layer1 = self._prepare_features(X, self._layer1)
        
        # Layer1: Anomaly detection
        health_status = self._layer1.predict(X_layer1)
        health_proba = self._layer1.predict_proba(X_layer1)
        
        results = pd.DataFrame({
            'health_status': health_status,
            'health_confidence': health_proba.max(axis=1),
            'warning_type': None,
            'failure_type': None
        })
        
        # Layer2: Warning classification
        if self._layer2_warning is not None:
            warning_mask = health_status == 'Warning'
            if warning_mask.any():
                X_warning = self._prepare_features(X[warning_mask], self._layer2_warning)
                warning_types = self._layer2_warning.predict(X_warning)
                results.loc[warning_mask, 'warning_type'] = warning_types
        
        # Layer2: Failure classification
        if self._layer2_failure is not None:
            failure_mask = health_status == 'Failure'
            if failure_mask.any():
                X_failure = self._prepare_features(X[failure_mask], self._layer2_failure)
                failure_types = self._layer2_failure.predict(X_failure)
                results.loc[failure_mask, 'failure_type'] = failure_types
        
        return results
    
    def _prepare_features(self, X: pd.DataFrame, model: XGBoostModel) -> np.ndarray:
        """Align features with model's expected feature names."""
        feature_names = model.get_feature_names()
        if feature_names is None:
            return X.select_dtypes(include=[np.number]).values
        
        # Select only features the model was trained on
        available_features = [f for f in feature_names if f in X.columns]
        return X[available_features].values

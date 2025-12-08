"""ML orchestrator for training and evaluation pipeline."""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.ML.core.interfaces import IFeatureExtractor, IModel

logger = logging.getLogger(__name__)


class MLOrchestrator:
    """Orchestrates ML training and evaluation workflow."""
    
    def __init__(
        self,
        model: IModel,
        target_col: str = 'health_status'
    ):
        """
        Args:
            model: ML/DL model implementation
            target_col: Name of target column
        """
        self._model = model
        self._target_col = target_col
    
    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        validation_df: pd.DataFrame = None,
        model_save_path: str = None
    ) -> dict:
        """
        Execute ML training and evaluation (features already extracted in preprocessing).
        
        Args:
            train_df: Training dataframe with features
            test_df: Test dataframe with features
            model_save_path: Path to save trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Features already extracted in preprocessing step
        X_train, y_train, feature_names = self._prepare_data(train_df)
        X_test, y_test, _ = self._prepare_data(test_df)

        # Set feature names in model for consistent inference
        self._model.set_feature_names(feature_names)

        print(f"Training model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self._model.train(X_train, y_train)

        print("Evaluating model...")
        metrics = self._model.evaluate(X_test, y_test)

        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")

        if model_save_path:
            print(f"Saving model to {model_save_path}...")
            self._model.save(model_save_path)
        
        return metrics
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Separate features and target, dropping specified columns."""
        drop_cols = ["timestamp", "health_status", "run_id"]
        if self._target_col not in df.columns:
            raise ValueError(f"Target column '{self._target_col}' not found in dataframe")

        # Drop target and any columns in drop_cols that exist in df
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        if self._target_col not in cols_to_drop:
            cols_to_drop = cols_to_drop + [self._target_col]

        feature_df = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
        feature_names = feature_df.columns.tolist()
        X = feature_df.values
        y = df[self._target_col].values

        return X, y, feature_names


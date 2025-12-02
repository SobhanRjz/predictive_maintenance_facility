"""ML orchestrator for training and evaluation pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from src.core.interfaces import IFeatureExtractor, IModel


class MLOrchestrator:
    """Orchestrates ML training and evaluation workflow."""
    
    def __init__(
        self,
        feature_extractor: IFeatureExtractor,
        model: IModel,
        target_col: str = 'health_status'
    ):
        """
        Args:
            feature_extractor: Feature extraction strategy
            model: ML/DL model implementation
            target_col: Name of target column
        """
        self._feature_extractor = feature_extractor
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
        Execute full ML pipeline: feature extraction, training, evaluation.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            model_save_path: Path to save trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Extracting features from training data...")
        train_features = self._feature_extractor.extract(train_df)
        print(f"Training features shape: {train_features.shape}")
        
        print("\nExtracting features from test data...")
        test_features = self._feature_extractor.extract(test_df)
        print(f"Test features shape: {test_features.shape}")
        
        # Prepare training data
        X_train, y_train = self._prepare_data(train_features)
        X_test, y_test = self._prepare_data(test_features)
        
        print(f"\nTraining model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        self._model.train(X_train, y_train)
        
        print("\nEvaluating model...")
        metrics = self._model.evaluate(X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")
        
        if model_save_path:
            print(f"\nSaving model to {model_save_path}...")
            self._model.save(model_save_path)
        
        return metrics
    
    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Separate features and target."""
        if self._target_col not in df.columns:
            raise ValueError(f"Target column '{self._target_col}' not found in dataframe")
        
        X = df.drop(columns=[self._target_col]).select_dtypes(include=[np.number]).values
        y = df[self._target_col].values
        
        return X, y


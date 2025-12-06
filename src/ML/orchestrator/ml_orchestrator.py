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
        logger.info("="*80)
        logger.info("ML TRAINING & EVALUATION")
        logger.info("="*80)
        
        # Features already extracted in preprocessing step
        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Check for run_id overlap (data leakage detection)
        if 'run_id' in train_df.columns and 'run_id' in test_df.columns:
            train_runs = set(train_df['run_id'].unique())
            test_runs = set(test_df['run_id'].unique())
            overlap = train_runs.intersection(test_runs)
            
            if overlap:
                logger.error(f"⚠️ DATA LEAKAGE DETECTED: {len(overlap)} run_ids in both train and test!")
                logger.error(f"Overlapping run_ids: {sorted(overlap)}")
                print(f"\n⚠️ WARNING: DATA LEAKAGE! {len(overlap)} run_ids appear in both train and test")
                print(f"Overlapping run_ids: {sorted(overlap)}")
            else:
                logger.info(f"✓ No data leakage: train and test run_ids are separate")
                print(f"✓ No data leakage: {len(train_runs)} train run_ids, {len(test_runs)} test run_ids (no overlap)")
        
        # Prepare training data
        X_train, y_train, feature_names = self._prepare_data(train_df)
        X_test, y_test, _ = self._prepare_data(test_df)

        logger.info(f"Features: {len(feature_names)} columns")
        logger.info(f"Feature names (first 10): {feature_names[:10]}")
        
        # Set feature names in model for consistent inference
        self._model.set_feature_names(feature_names)
        
        print(f"\nTraining model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        logger.info(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        self._model.train(X_train, y_train)
        
        print("\nEvaluating model...")
        logger.info("Evaluating model on test set...")
        metrics = self._model.evaluate(X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        
        if model_save_path:
            print(f"\nSaving model to {model_save_path}...")
            logger.info(f"Saving model to {model_save_path}")
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


"""Inference module for ML model predictions."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .features.time_domain_features import TimeDomainFeatureExtractor
from .models.xgboost_model import XGBoostModel


class MLInference:
    """Handles ML model inference for real-time predictions."""

    def __init__(
        self,
        model_path: str = "models/xgboost_model.pkl",
        window_size: str = "10min",
        timestamp_col: str = "timestamp",
        target_col: str = "health_status",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ML inference.

        Args:
            model_path: Path to saved model file
            window_size: Time window for feature extraction
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._model_path = Path(model_path)

        # Initialize feature extractor
        self._feature_extractor = TimeDomainFeatureExtractor(
            window_size=window_size,
            timestamp_col=timestamp_col,
            target_col=target_col
        )

        # Initialize model
        self._model = XGBoostModel()
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from disk."""
        try:
            if not self._model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            self._model.load(str(self._model_path))
            self.logger.info(f"ML model loaded successfully from {self._model_path}")

            # Sanity check: print label encoder classes
            if hasattr(self._model, '_label_encoder') and self._model._label_encoder.classes_.size > 0:
                self.logger.info(f"Model classes: {list(self._model._label_encoder.classes_)}")
                print(f"DEBUG - Model classes: {list(self._model._label_encoder.classes_)}")
            else:
                self.logger.warning("Label encoder classes not available")

        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            raise

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions from raw sensor data batch.

        Args:
            df: DataFrame with columns: timestamp, sensor readings, health_status (optional)

        Returns:
            DataFrame with predictions and probabilities
        """
        try:
            self.logger.debug(f"Processing batch with {len(df)} rows")

            # Extract features
            features_df = self._feature_extractor.extract(df)

            # Prepare features for prediction using stored feature names
            feature_names = self._model.get_feature_names()
            if feature_names is None:
                raise RuntimeError("Model was not trained with feature names. Retrain the model to enable inference.")
            X = features_df[feature_names].values

            # Make predictions
            predictions = self._model.predict(X)
            probabilities = self._model.predict_proba(X)

            # Get class names
            class_names = self._model._label_encoder.classes_

            # Create results DataFrame
            results = features_df[['timestamp']].copy()
            results['predicted_health_status'] = predictions

            # Add probability columns
            for i, class_name in enumerate(class_names):
                results[f'prob_{class_name}'] = probabilities[:, i]

            self.logger.debug(f"Batch prediction completed for {len(results)} samples")
            return results

        except Exception as e:
            self.logger.error(f"Error during batch prediction: {e}")
            raise

    def predict_single(self, row: pd.Series) -> dict:
        """
        Make prediction for a single row.

        Args:
            row: Single row of sensor data

        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame([row])

            # Get predictions
            results_df = self.predict_batch(df)
            result = results_df.iloc[0]

            # Extract prediction details
            predicted_status = result['predicted_health_status']

            # Get probabilities
            prob_cols = [col for col in results_df.columns if col.startswith('prob_')]
            probabilities = {col.replace('prob_', ''): result[col] for col in prob_cols}
            max_prob = max(probabilities.values())

            return {
                'predicted_status': predicted_status,
                'probabilities': probabilities,
                'confidence': max_prob,
                'timestamp': result['timestamp']
            }

        except Exception as e:
            self.logger.error(f"Error during single prediction: {e}")
            raise
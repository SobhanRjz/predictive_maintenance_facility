"""Factory for creating ML orchestrator with dependencies."""
from src.orchestrator.ml_orchestrator import MLOrchestrator
from src.features.time_domain_features import TimeDomainFeatureExtractor
from src.models.xgboost_model import XGBoostModel


class MLOrchestratorFactory:
    """Factory for creating configured ML orchestrator."""
    
    @staticmethod
    def create(
        feature_window_size: str = '10T',
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        target_col: str = 'health_status',
        random_state: int = 42,
        early_stopping_rounds: int = None
    ) -> MLOrchestrator:
        """
        Create ML orchestrator with dependencies.

        Args:
            feature_window_size: Time window for feature extraction
            n_estimators: XGBoost number of trees
            max_depth: XGBoost max tree depth
            learning_rate: XGBoost learning rate
            target_col: Target column name
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds (None to disable)
        """
        feature_extractor = TimeDomainFeatureExtractor(
            window_size=feature_window_size,
            target_col=target_col
        )

        model = XGBoostModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            target_col=target_col,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds
        )

        return MLOrchestrator(
            feature_extractor=feature_extractor,
            model=model,
            target_col=target_col
        )


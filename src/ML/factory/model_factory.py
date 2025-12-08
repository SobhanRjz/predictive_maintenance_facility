"""Generic model factory supporting XGBoost with config-driven instantiation."""
from src.ML.core.interfaces import IModel
from src.ML.config.config_loader import ModelConfig


class ModelFactory:
    """Creates models based on configuration."""
    
    @staticmethod
    def create(config: ModelConfig) -> IModel:
        """
        Create model instance from configuration.
        
        Args:
            config: Parsed model configuration
            
        Returns:
            Model instance implementing IModel
        """
        if config.type.lower() == "xgboost":
            from src.ML.models.xgboost_model import XGBoostModel
            
            return XGBoostModel(
                n_estimators=config.hyperparameters.get('n_estimators', 100),
                max_depth=config.hyperparameters.get('max_depth', 6),
                learning_rate=config.hyperparameters.get('learning_rate', 0.1),
                target_col=config.target.column,
                random_state=config.hyperparameters.get('random_state', 42),
                early_stopping_rounds=config.hyperparameters.get('early_stopping_rounds'),
                eval_metric=config.hyperparameters.get('eval_metric', 'mlogloss'),
                scale_pos_weight=config.hyperparameters.get('scale_pos_weight')
            )
        else:
            raise ValueError(f"Unsupported model type: {config.type}")

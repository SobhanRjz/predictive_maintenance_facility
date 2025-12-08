"""Main script for complete ML pipeline: preprocessing + training."""
import logging
import sys
from pathlib import Path
from src.ML.config.config_loader import ConfigLoader, ModelConfig
from src.ML.factory.orchestrator_factory import OrchestratorFactory
from src.ML.factory.model_factory import ModelFactory
from src.ML.orchestrator.ml_orchestrator import MLOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Execute complete ML pipeline from layer config."""
    # Load configuration from layer config YAML
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/model_configs/layer1_anomaly_detection.yaml"

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("\nUsage: python main_ml_pipeline.py [config_path]")
        print("Default: python main_ml_pipeline.py configs/model_configs/layer1_anomaly_detection.yaml")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")

    config: ModelConfig = ConfigLoader.load(config_path)
    dataset_config = config.dataset
    features_config = config.features
    
    preprocessing_orchestrator = OrchestratorFactory.create(
        use_timeseries_split=dataset_config.use_timeseries_split,
        use_stratification=dataset_config.use_stratification,
        window_size=features_config.window_size,
        resample_freq=dataset_config.resample_freq,
        resample_method=dataset_config.resample_method,
        extract_features=features_config.extract_features,
        timestamp_col=features_config.timestamp_col,
        target_col=config.target.column,
        exclude_cols=features_config.exclude_cols
    )
    
    train_df, test_df, validation_report = preprocessing_orchestrator.process(
        normal_paths=dataset_config.get_normal_paths(),
        abnormal_paths=dataset_config.get_abnormal_paths(),
        output_dir=dataset_config.output_dir,
        test_size=dataset_config.test_size,
        random_state=dataset_config.random_state,
        add_hierarchical_labels=dataset_config.add_hierarchical_labels
    )

    # Print class distribution statistics
    train_counts = train_df[config.target.column].value_counts()
    test_counts = test_df[config.target.column].value_counts()

    print(f"Dataset: {len(train_df)} train, {len(test_df)} test samples")

    # Step 2: ML Training & Evaluation
    print("\nTraining model...")
    
    # Create model from config
    model = ModelFactory.create(config)
    
    # Create orchestrator
    ml_orchestrator = MLOrchestrator(
        model=model,
        target_col=config.target.column
    )
    
    model_path = config.output.model_path
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics = ml_orchestrator.train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        model_save_path=model_path
    )
    
    # Save metrics
    import json
    metrics_path = config.output.metrics_path
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics_serializable = {
        k: v for k, v in metrics.items() 
        if k != 'classification_report'
    }
    metrics_serializable['classification_report_text'] = metrics['classification_report']
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    logger.info("Pipeline completed successfully")


if __name__ == '__main__':
    main()

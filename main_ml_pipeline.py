"""Main script for complete ML pipeline: preprocessing + training."""
from src.config.config import DatasetConfig, MLConfig
from src.factory.orchestrator_factory import OrchestratorFactory
from src.factory.ml_orchestrator_factory import MLOrchestratorFactory


def main():
    """Execute complete ML pipeline."""
    # Configuration
    dataset_config = DatasetConfig()
    ml_config = MLConfig()
    
    # Step 1: Data Preprocessing
    print("=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    preprocessing_orchestrator = OrchestratorFactory.create(
        use_timeseries_split=dataset_config.use_timeseries_split,
        use_stratification=dataset_config.use_stratification,
        window_size=dataset_config.window_size,
        resample_freq=dataset_config.resample_freq,
        resample_method=dataset_config.resample_method,
        extract_features=True  # Extract features before splitting
    )
    
    train_df, test_df, validation_report = preprocessing_orchestrator.process(
        normal_paths=dataset_config.get_normal_paths(),
        abnormal_paths=dataset_config.get_abnormal_paths(),
        output_dir=dataset_config.output_dir,
        test_size=dataset_config.test_size,
        random_state=dataset_config.random_state
    )

    # Print class distribution statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    train_counts = train_df['health_status'].value_counts()
    test_counts = test_df['health_status'].value_counts()

    print(f"Train set total: {len(train_df)} samples")
    print(f"  Normal: {train_counts.get('Normal', 0)}")
    print(f"  Warning: {train_counts.get('Warning', 0)}")
    print(f"  Failure: {train_counts.get('Failure', 0)}")

    print(f"\nTest set total: {len(test_df)} samples")
    print(f"  Normal: {test_counts.get('Normal', 0)}")
    print(f"  Warning: {test_counts.get('Warning', 0)}")
    print(f"  Failure: {test_counts.get('Failure', 0)}")

    print(f"\nOverall total: {len(train_df) + len(test_df)} samples")

    # Step 2: ML Training & Evaluation
    print("\n" + "=" * 80)
    print("STEP 2: ML TRAINING & EVALUATION")
    print("=" * 80)
    
    ml_orchestrator = MLOrchestratorFactory.create(
        feature_window_size=ml_config.feature_window_size,
        n_estimators=ml_config.n_estimators,
        max_depth=ml_config.max_depth,
        learning_rate=ml_config.learning_rate,
        target_col=ml_config.target_col,
        random_state=ml_config.random_state,
        early_stopping_rounds=ml_config.early_stopping_rounds
    )
    
    model_path = f"{ml_config.model_output_dir}/xgboost_model.pkl"
    metrics = ml_orchestrator.train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        model_save_path=model_path
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    main()


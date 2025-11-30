"""Main entry point for preprocessing pipeline."""
import json
from src.factory.orchestrator_factory import OrchestratorFactory
from src.config.config import DatasetConfig


def main():
    """Execute preprocessing pipeline."""
    print("="*60)
    print("Motor Sensor Data Preprocessing Pipeline")
    print("="*60)
    
    config = DatasetConfig()
    orchestrator = OrchestratorFactory.create(
        use_timeseries_split=config.use_timeseries_split,
        use_stratification=config.use_stratification,
        window_size=config.window_size,
        resample_freq=config.resample_freq,
        resample_method=config.resample_method
    )
    
    train_df, test_df, validation_report = orchestrator.process(
        normal_paths=config.get_normal_paths(),
        abnormal_paths=config.get_abnormal_paths(),
        output_dir=config.output_dir,
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"\nTrain set distribution:\n{train_df['health_status'].value_counts()}")
    print(f"\nTest set distribution:\n{test_df['health_status'].value_counts()}")
    
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Total rows: {validation_report['total_rows']}")
    print(f"Missing values: {validation_report['missing_values']['total_missing']}")
    print(f"Duplicate timestamps: {validation_report['duplicate_timestamps']}")
    print(f"Timestamps sorted: {validation_report['timestamp_sorted']}")
    print(f"Temporal gaps detected: {validation_report['temporal_gaps']['gaps_detected']}")
    print(f"Median time interval: {validation_report['temporal_gaps']['median_interval']}")
    
    if validation_report['outliers']:
        print(f"\nOutliers detected in {len(validation_report['outliers'])} columns")
    
    # Save validation report
    with open(f"{config.output_dir}/validation_report.json", 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"\nData exported to: {config.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()


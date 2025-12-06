"""Main orchestrator for preprocessing pipeline."""
import pandas as pd
from pathlib import Path
from src.ML.core.interfaces import IDataFilter, IDataSplitter, IDataExporter, IFeatureExtractor
from src.ML.loaders.data_loader import MultiFileLoader
from src.ML.validators.data_validator import DataValidator


class PreprocessingOrchestrator:
    """Orchestrates the complete preprocessing workflow."""
    
    def __init__(
        self,
        loader: MultiFileLoader,
        normal_filter: IDataFilter,
        abnormal_filter: IDataFilter,
        splitter: IDataSplitter,
        exporter: IDataExporter,
        validator: DataValidator,
        feature_extractor: IFeatureExtractor = None
    ):
        self._loader = loader
        self._normal_filter = normal_filter
        self._abnormal_filter = abnormal_filter
        self._splitter = splitter
        self._exporter = exporter
        self._validator = validator
        self._feature_extractor = feature_extractor
    
    def process(
        self,
        normal_paths: list[str],
        abnormal_paths: list[str],
        output_dir: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Execute full preprocessing pipeline.
        
        Args:
            normal_paths: Paths to normal condition datasets
            abnormal_paths: Paths to failure/warning datasets
            output_dir: Directory for output files
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df, validation_report)
        """
        print("Loading normal condition data...")
        normal_df = self._loader.load_multiple(normal_paths, start_run_id=1)
        normal_df = self._normal_filter.filter(normal_df)
        print(f"Normal data loaded: {len(normal_df)} rows")

        print("\nLoading failure/warning data...")
        next_run_id = len(normal_paths) + 1
        abnormal_df = self._loader.load_multiple(abnormal_paths, start_run_id=next_run_id)
        # abnormal_df = self._abnormal_filter.filter(abnormal_df)
        print(f"Abnormal data loaded: {len(abnormal_df)} rows")
        
        # Combine datasets
        print("\nCombining datasets...")
        combined_df = pd.concat([normal_df, abnormal_df], ignore_index=True)
        
        # Validate data quality
        print("\nValidating data quality...")
        is_valid, validation_report = self._validator.validate(combined_df)
        print(f"Data validation: {'PASSED' if is_valid else 'WARNINGS DETECTED'}")
        
        # Extract features BEFORE splitting (correct ML pipeline order)
        if self._feature_extractor:
            print("\nExtracting features from combined data...")
            combined_df = self._feature_extractor.extract(combined_df)
            print(f"Features extracted: {combined_df.shape}")
        
        
        # Split data
        print("\nSplitting data...")
        train_df, test_df = self._splitter.split(combined_df, test_size, random_state)
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Export
        print(f"\nExporting to {output_dir}...")
        self._exporter.export(train_df, test_df, output_dir)
        
        return train_df, test_df, validation_report


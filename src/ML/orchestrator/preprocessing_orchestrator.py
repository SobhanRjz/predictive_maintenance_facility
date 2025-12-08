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
        random_state: int = 42,
        add_hierarchical_labels: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Execute full preprocessing pipeline.
        
        Args:
            normal_paths: Paths to normal condition datasets
            abnormal_paths: Paths to failure/warning datasets
            output_dir: Directory for output files
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            add_hierarchical_labels: Add warning_type and failure_type columns
            
        Returns:
            Tuple of (train_df, test_df, validation_report)
        """
        if len(normal_paths) > 0:
            normal_df = self._loader.load_multiple(
                normal_paths,
                start_run_id=1,
                add_hierarchical_labels=add_hierarchical_labels
            )
            normal_df = self._normal_filter.filter(normal_df)
        else:
            normal_df = pd.DataFrame()

        next_run_id = len(normal_df) + 1
        if len(abnormal_paths) > 0:
            abnormal_df = self._loader.load_multiple(
                abnormal_paths,
                start_run_id=next_run_id,
                add_hierarchical_labels=add_hierarchical_labels
            )
            abnormal_df = self._abnormal_filter.filter(abnormal_df)
        else:
            abnormal_df = pd.DataFrame()

        # Combine datasets
        combined_df = pd.concat([normal_df, abnormal_df], ignore_index=True)

        # Validate data quality
        is_valid, validation_report = self._validator.validate(combined_df)

        # Extract features BEFORE splitting (correct ML pipeline order)
        if self._feature_extractor:
            combined_df0 = self._feature_extractor.extract(combined_df)
            if 'health_status' not in combined_df0.columns:
                combined_df0['health_status'] = combined_df['health_status']
            combined_df = combined_df0

        # Split data
        train_df, test_df = self._splitter.split(combined_df, test_size, random_state)
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Export
        self._exporter.export(train_df, test_df, output_dir)
        
        return train_df, test_df, validation_report


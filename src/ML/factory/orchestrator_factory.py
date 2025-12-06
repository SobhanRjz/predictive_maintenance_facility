"""Factory for creating orchestrator with dependencies."""
from src.ML.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator
from src.ML.loaders.data_loader import CSVLoader, ExcelLoader, MultiFileLoader, DataResampler
from src.ML.filters.condition_filter import ExcludeNormalFilter, IncludeOnlyNormalFilter
from src.ML.splitters.timeseries_splitter import TimeSeriesSplitter, StratifiedTimeSeriesSplitter
from src.ML.exporters.data_exporter import CSVExporter
from src.ML.validators.data_validator import DataValidator
from src.ML.features.time_domain_features import TimeDomainFeatureExtractor
from src.ML.splitters.data_splitter import RunIdSplitter

class OrchestratorFactory:
    """Factory for creating configured preprocessing orchestrator."""

    @staticmethod
    def create(
        use_timeseries_split: bool = True,
        use_stratification: bool = True,
        window_size: str = '10min',
        resample_freq: str = '1T',
        resample_method: str = 'mean',
        extract_features: bool = True
    ) -> PreprocessingOrchestrator:
        """
        Create orchestrator with dependencies.

        Args:
            use_timeseries_split: If True, uses chronological split
            use_stratification: If True with use_timeseries_split, uses stratified time-series split
            window_size: Window size for feature extraction and splitting (e.g., '10min', '1H')
            resample_freq: Resampling frequency (e.g., '1T' for 1 minute)
            resample_method: Resampling aggregation method ('mean', 'max', 'min', 'first', 'last')
            extract_features: If True, extract features before splitting (recommended)
        """
        # Create resampler for 30sec -> 1min aggregation
        resampler = DataResampler(freq=resample_freq, agg_method=resample_method)

        csv_loader = CSVLoader()
        excel_loader = ExcelLoader()
        multi_loader = MultiFileLoader(csv_loader, excel_loader, resampler=resampler)

        normal_filter = IncludeOnlyNormalFilter()
        abnormal_filter = ExcludeNormalFilter()

        # Use run_id based splitting for proper train/test separation by CSV file
        
        splitter = RunIdSplitter()
        
        # Feature extractor (applied before splitting)
        feature_extractor = TimeDomainFeatureExtractor(
            window_size=window_size,
            timestamp_col='timestamp',
            target_col='health_status',
            exclude_cols=['run_id']
        ) if extract_features else None
        
        exporter = CSVExporter()
        validator = DataValidator()

        return PreprocessingOrchestrator(
            loader=multi_loader,
            normal_filter=normal_filter,
            abnormal_filter=abnormal_filter,
            splitter=splitter,
            exporter=exporter,
            validator=validator,
            feature_extractor=feature_extractor
        )


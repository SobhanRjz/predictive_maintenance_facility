"""Factory for creating orchestrator with dependencies."""
from src.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator
from src.loaders.data_loader import CSVLoader, ExcelLoader, MultiFileLoader, DataResampler
from src.filters.condition_filter import ExcludeNormalFilter, IncludeOnlyNormalFilter
from src.splitters.timeseries_splitter import TimeSeriesSplitter, StratifiedTimeSeriesSplitter
from src.exporters.data_exporter import CSVExporter
from src.validators.data_validator import DataValidator


class OrchestratorFactory:
    """Factory for creating configured preprocessing orchestrator."""

    @staticmethod
    def create(
        use_timeseries_split: bool = True,
        use_stratification: bool = True,
        window_size: str = '10T',
        resample_freq: str = '1T',
        resample_method: str = 'mean'
    ) -> PreprocessingOrchestrator:
        """
        Create orchestrator with dependencies.

        Args:
            use_timeseries_split: If True, uses chronological split
            use_stratification: If True with use_timeseries_split, uses stratified time-series split
            window_size: Window size for stratified splitting (e.g., '10T', '1H', '1D')
            resample_freq: Resampling frequency (e.g., '1T' for 1 minute)
            resample_method: Resampling aggregation method ('mean', 'max', 'min', 'first', 'last')
        """
        # Create resampler for 30sec -> 1min aggregation
        resampler = DataResampler(freq=resample_freq, agg_method=resample_method)

        csv_loader = CSVLoader()
        excel_loader = ExcelLoader()
        multi_loader = MultiFileLoader(csv_loader, excel_loader, resampler=resampler)

        normal_filter = IncludeOnlyNormalFilter()
        abnormal_filter = ExcludeNormalFilter()

        # if use_timeseries_split and use_stratification:
        #     splitter = StratifiedTimeSeriesSplitter(window_size=window_size)
        # elif use_timeseries_split:
        #     splitter = TimeSeriesSplitter()
        # else:
        #     from src.splitters.data_splitter import StratifiedSplitter
        #     splitter = StratifiedSplitter()

                # Use run_id based splitting instead of time-series or stratified
        from src.splitters.data_splitter import RunIdSplitter
        splitter = RunIdSplitter()
        
        exporter = CSVExporter()
        validator = DataValidator()

        return PreprocessingOrchestrator(
            loader=multi_loader,
            normal_filter=normal_filter,
            abnormal_filter=abnormal_filter,
            splitter=splitter,
            exporter=exporter,
            validator=validator
        )


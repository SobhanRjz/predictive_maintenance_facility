"""Data loader with config-driven filtering for hierarchical models."""
import pandas as pd
import logging
from pathlib import Path
from src.ML.config.config_loader import DataFilterConfig

logger = logging.getLogger(__name__)


class FilteredDataLoader:
    """Loads and filters data based on configuration."""
    
    @staticmethod
    def load_and_filter(
        train_path: str | Path,
        test_path: str | Path,
        data_filter: DataFilterConfig = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed data and apply filtering.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            data_filter: Optional filter configuration
            
        Returns:
            Filtered (train_df, test_df)
        """
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        
        if data_filter is not None:
            train_df = FilteredDataLoader._apply_filter(train_df, data_filter)
            test_df = FilteredDataLoader._apply_filter(test_df, data_filter)
            logger.info(f"After filter - train: {train_df.shape}, test: {test_df.shape}")
        
        return train_df, test_df
    
    @staticmethod
    def _apply_filter(df: pd.DataFrame, filter_config: DataFilterConfig) -> pd.DataFrame:
        """Apply single filter condition to dataframe."""
        col = filter_config.column
        op = filter_config.operator
        val = filter_config.value
        
        if col not in df.columns:
            raise ValueError(f"Filter column '{col}' not found in dataframe")
        
        if op == "==":
            return df[df[col] == val].copy()
        elif op == "!=":
            return df[df[col] != val].copy()
        elif op == "in":
            return df[df[col].isin(val)].copy()
        else:
            raise ValueError(f"Unsupported filter operator: {op}")

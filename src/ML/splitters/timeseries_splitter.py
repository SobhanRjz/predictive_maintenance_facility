"""Time-series aware data splitting with stratification support."""
import pandas as pd
from collections import Counter
from typing import Tuple
from src.core.interfaces import IDataSplitter


class TimeSeriesSplitter(IDataSplitter):
    """Splits time-series data chronologically without shuffling."""
    
    def __init__(self, timestamp_col: str = 'timestamp'):
        self._timestamp_col = timestamp_col
    
    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data chronologically.
        
        Args:
            df: DataFrame with timestamp column
            test_size: Proportion for test set
            random_state: Unused (for interface compatibility)
            
        Returns:
            Tuple of (train_df, test_df) split chronologically
        """
        # Sort by timestamp
        df_sorted = df.sort_values(by=self._timestamp_col).reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        return train_df, test_df


class StratifiedTimeSeriesSplitter(IDataSplitter):
    """
    Splits time-series data while maintaining class distribution per time window.
    
    Strategy:
    - Preserves temporal order (test is contiguous suffix in time)
    - Approximates stratification by selecting whole time windows
    - Works backwards from latest window until target class counts met
    """
    
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        stratify_col: str = "health_status",
        window_size: str = "1H",
    ):
        self._timestamp_col = timestamp_col
        self._stratify_col = stratify_col
        self._window_size = window_size
    
    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time-series while preserving temporal order and approximating stratification.
        
        Args:
            df: DataFrame with timestamp and stratification columns
            test_size: Proportion for test set (0.0 to 1.0)
            random_state: Kept for API compatibility, not used
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if self._timestamp_col not in df.columns:
            raise KeyError(f"timestamp column '{self._timestamp_col}' not found in df.")
        if self._stratify_col not in df.columns:
            raise KeyError(f"stratify column '{self._stratify_col}' not found in df.")
        
        # Sort by timestamp once
        df_sorted = df.sort_values(by=self._timestamp_col).reset_index(drop=True)
        df_sorted[self._timestamp_col] = pd.to_datetime(df_sorted[self._timestamp_col])
        
        # Create time windows
        df_sorted["_time_window"] = df_sorted[self._timestamp_col].dt.floor(self._window_size)
        
        # Pre-compute window-class counts (vectorized)
        window_class_counts = df_sorted.groupby(["_time_window", self._stratify_col]).size().unstack(fill_value=0)
        
        # Global class counts
        global_counts = df_sorted[self._stratify_col].value_counts().to_dict()
        if not global_counts:
            return df_sorted.drop(columns=["_time_window"]).copy(), df_sorted.iloc[0:0].copy()
        
        # Target test counts per class
        target_test_counts = {cls: max(1, int(cnt * test_size)) for cls, cnt in global_counts.items()}
        
        # Get sorted windows (latest first for reverse iteration)
        windows = sorted(window_class_counts.index, reverse=True)
        
        # Accumulate test counts efficiently
        total_rows = len(df_sorted)
        min_test_rows = max(1, int(total_rows * test_size))
        
        test_counts = {cls: 0 for cls in global_counts.keys()}
        test_window_idx = 0
        
        # Iterate windows from latest to oldest
        for test_window_idx, w in enumerate(windows, start=1):
            # Add current window's class counts
            for cls in global_counts.keys():
                if cls in window_class_counts.columns:
                    test_counts[cls] += window_class_counts.loc[w, cls]
            
            # Check stopping conditions
            enough_per_class = all(test_counts[cls] >= target_test_counts[cls] for cls in global_counts.keys())
            
            if enough_per_class:
                # Compute actual test rows
                test_windows_set = set(windows[:test_window_idx])
                test_rows = df_sorted["_time_window"].isin(test_windows_set).sum()
                
                if test_rows >= min_test_rows:
                    break
        
        # Final split using boolean mask (faster than multiple operations)
        test_windows_final = set(windows[:test_window_idx])
        test_mask = df_sorted["_time_window"].isin(test_windows_final)
        
        train_df = df_sorted.loc[~test_mask].drop(columns=["_time_window"]).copy()
        test_df = df_sorted.loc[test_mask].drop(columns=["_time_window"]).copy()
        
        return train_df, test_df

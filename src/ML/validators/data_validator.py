"""Data quality validation for time-series motor data."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataValidator:
    """Validates data quality and integrity."""
    
    def __init__(self, timestamp_col: str = 'timestamp'):
        self._timestamp_col = timestamp_col
    
    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, any]]:
        """
        Comprehensive validation of dataset.
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'total_rows': len(df),
            'missing_values': self._check_missing(df),
            'duplicate_timestamps': self._check_duplicates(df),
            'timestamp_sorted': self._check_sorted(df),
            'temporal_gaps': self._check_temporal_gaps(df),
            'numeric_ranges': self._check_numeric_ranges(df),
            'outliers': self._check_outliers(df)
        }
        
        is_valid = (
            report['missing_values']['total_missing'] == 0 and
            report['duplicate_timestamps'] == 0 and
            report['timestamp_sorted']
        )
        
        return is_valid, report
    
    def _check_missing(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for missing values."""
        missing_per_col = df.isnull().sum()
        return {
            'total_missing': missing_per_col.sum(),
            'columns_with_missing': missing_per_col[missing_per_col > 0].to_dict()
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """Check for duplicate timestamps."""
        if self._timestamp_col in df.columns:
            return df[self._timestamp_col].duplicated().sum()
        return 0
    
    def _check_sorted(self, df: pd.DataFrame) -> bool:
        """Check if timestamps are sorted."""
        if self._timestamp_col in df.columns:
            timestamps = pd.to_datetime(df[self._timestamp_col])
            return timestamps.is_monotonic_increasing
        return True
    
    def _check_temporal_gaps(self, df: pd.DataFrame) -> Dict[str, any]:
        """Detect unusual gaps in time series."""
        if self._timestamp_col not in df.columns:
            return {'gaps_detected': 0}
        
        timestamps = pd.to_datetime(df[self._timestamp_col])
        time_diffs = timestamps.diff()
        median_diff = time_diffs.median()
        
        # Gap threshold: 3x median interval
        large_gaps = time_diffs[time_diffs > median_diff * 3]
        
        return {
            'gaps_detected': len(large_gaps),
            'median_interval': str(median_diff),
            'max_gap': str(time_diffs.max()) if len(time_diffs) > 0 else None
        }
    
    def _check_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Check numeric column ranges."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ranges = {}
        for col in numeric_cols:
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        return ranges
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
        
        return outlier_counts


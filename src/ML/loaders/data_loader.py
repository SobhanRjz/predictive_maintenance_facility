"""Data loader implementations."""
from pathlib import Path
import pandas as pd
import numpy as np
from src.core.interfaces import IDataLoader
from concurrent.futures import ThreadPoolExecutor

class CSVLoader(IDataLoader):
    """Loads CSV files."""

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df


class ExcelLoader(IDataLoader):
    """Loads Excel files."""

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_excel(path)
        return df


class DataResampler:
    """Resamples time-series data to specified frequency."""

    def __init__(self, freq: str = '1min', agg_method: str = 'mean'):
        """
        Args:
            freq: Pandas frequency string (e.g., '1min' for 1 minute)
            agg_method: Aggregation method ('mean', 'max', 'min', 'first', 'last')
        """
        self._freq = freq
        self._agg_method = agg_method

    def resample(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Resample dataframe to specified frequency."""
        if timestamp_col not in df.columns:
            return df

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # Resample numeric columns
        if not numeric_cols.empty:
            if self._agg_method == 'mean':
                df_resampled = df[numeric_cols].resample(self._freq).mean()
            elif self._agg_method == 'max':
                df_resampled = df[numeric_cols].resample(self._freq).max()
            elif self._agg_method == 'min':
                df_resampled = df[numeric_cols].resample(self._freq).min()
            elif self._agg_method == 'first':
                df_resampled = df[numeric_cols].resample(self._freq).first()
            elif self._agg_method == 'last':
                df_resampled = df[numeric_cols].resample(self._freq).last()
            else:
                raise ValueError(f"Unsupported aggregation method: {self._agg_method}")

            # Handle categorical columns (take first value)
            if not categorical_cols.empty:
                df_cat = df[categorical_cols].resample(self._freq).first()
                df_resampled = pd.concat([df_resampled, df_cat], axis=1)

        else:
            # No numeric columns, just resample categorical
            df_resampled = df.resample(self._freq).first()

        # Reset index and remove empty rows
        df_resampled = df_resampled.dropna(how='all').reset_index()

        return df_resampled


class MultiFileLoader:
    """Loads multiple files using appropriate loader."""

    def __init__(self, csv_loader: IDataLoader, excel_loader: IDataLoader, resampler: DataResampler = None):
        self._csv_loader = csv_loader
        self._excel_loader = excel_loader
        self._resampler = resampler

    def clean_health_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix truncated health status values."""
        df = df.copy()
        if 'health_status' in df.columns:
            df['health_status'] = df['health_status'].str.strip()  # Remove any whitespace
            df['health_status'] = df['health_status'].replace({
                'Warnin': 'Warning',
                'Failur': 'Failure',
                'Normal': 'Normal'  # Already correct
            })
        return df
    

    def _load_single(self, path: str) -> pd.DataFrame:
        import os
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        loader = self._excel_loader if ext in ('.xlsx', '.xls') else self._csv_loader
        df = loader.load(path)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = self.clean_health_status(df)
            if self._resampler is not None:
                df = self._resampler.resample(df)
        return df

    def load_multiple(self, paths: list[str], start_run_id: int = 1) -> pd.DataFrame:
        """Load multiple files with sequential run_id assignment."""
        if not paths:
            return pd.DataFrame()

        dfs_with_run_ids = []
        current_run_id = start_run_id
        for path in paths:
            df = self._load_single(path)
            df['run_id'] = current_run_id  # Add run_id column
            dfs_with_run_ids.append(df)
            current_run_id += 1

        return pd.concat(dfs_with_run_ids, ignore_index=True)

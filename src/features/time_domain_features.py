"""Time domain feature extraction for sensor data."""
import pandas as pd
import numpy as np
from src.core.interfaces import IFeatureExtractor


class TimeDomainFeatureExtractor(IFeatureExtractor):
    """Extracts statistical time-domain features from sensor signals."""
    
    def __init__(
        self,
        window_size: str = '10T',
        timestamp_col: str = 'timestamp',
        target_col: str = 'health_status',
        exclude_cols: list[str] = None
    ):
        """
        Args:
            window_size: Time window for feature extraction (e.g., '10T', '1H')
            timestamp_col: Name of timestamp column
            target_col: Name of target column to preserve
            exclude_cols: Additional columns to exclude from feature extraction
        """
        self._window_size = window_size
        self._timestamp_col = timestamp_col
        self._target_col = target_col
        self._exclude_cols = exclude_cols or ["run_id"]
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast time-domain feature extraction using vectorized groupby ops.
        """
        df = df.copy()
        df[self._timestamp_col] = pd.to_datetime(df[self._timestamp_col])

        # Identify sensor columns (numeric, excluding timestamp and target)
        exclude_set = {self._timestamp_col, self._target_col, *self._exclude_cols}
        sensor_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_set
        ]

        # Group by time windows
        df["_window"] = df[self._timestamp_col].dt.floor(self._window_size)

        # --- 1) Basic stats (vectorized) ---
        g = df.groupby("_window")[sensor_cols]

        means = g.mean()
        stds = g.std(ddof=0)           # match np.std default
        mins = g.min()
        maxs = g.max()

        # --- 2) RMS (sqrt of mean of squares) ---
        df_sq = df.copy()
        df_sq[sensor_cols] = df_sq[sensor_cols] ** 2
        rms = df_sq.groupby("_window")[sensor_cols].mean()
        rms = rms.apply(np.sqrt)

        # --- 3) Peak-to-peak (max - min) ---
        peak_to_peak = maxs - mins

        # --- 4) Skewness & kurtosis (vectorized) ---
        skew = g.skew()
        kurt = g.apply(lambda x: x.kurt())  # DataFrameGroupBy.kurt often exists, but this is safe

        # --- 5) Combine everything into one wide feature table ---
        # Build a multi-index columns DataFrame then flatten
        stats_dict = {
            "mean": means,
            "std": stds,
            "min": mins,
            "max": maxs,
            "rms": rms,
            "peak_to_peak": peak_to_peak,
            "skewness": skew,
            "kurtosis": kurt,
        }

        features = pd.concat(stats_dict, axis=1)  # columns: (stat, sensor)

        # Flatten MultiIndex → col_stat
        features.columns = [
            f"{sensor}_{stat}" for stat, sensor in features.columns.to_flat_index()
        ]

        # --- 6) Target: majority vote per window (like before) ---
        if self._target_col in df.columns:
            target = (
                df.groupby("_window")[self._target_col]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            )
            features[self._target_col] = target

        # --- 7) Replace _window with timestamp col name and reset index ---
        features = features.reset_index().rename(columns={"_window": self._timestamp_col})

        return features

    
    def _compute_features(self, col_name: str, values: np.ndarray) -> dict:
        """Compute time-domain features for a single sensor."""
        return {
            f'{col_name}_mean': np.mean(values),
            f'{col_name}_std': np.std(values),
            f'{col_name}_min': np.min(values),
            f'{col_name}_max': np.max(values),
            f'{col_name}_rms': np.sqrt(np.mean(values**2)),
            f'{col_name}_peak_to_peak': np.ptp(values),
            f'{col_name}_kurtosis': self._kurtosis(values),
            f'{col_name}_skewness': self._skewness(values)
        }
    
    @staticmethod
    def _kurtosis(values: np.ndarray) -> float:
        """Calculate kurtosis (fourth standardized moment)."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 4)
    
    @staticmethod
    def _skewness(values: np.ndarray) -> float:
        """Calculate skewness (third standardized moment)."""
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)


"""Data splitting implementations."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ML.core.interfaces import IDataSplitter


class RunIdSplitter(IDataSplitter):
    """Splits data by run_id: 80% of each run_id for training, 20% for testing."""

    def __init__(self, run_id_column: str = 'run_id'):
        self._run_id_column = run_id_column

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by run_id: take test_size percentage from each run_id for testing.
        Maintains temporal order within each run_id.
        """
        if self._run_id_column not in df.columns:
            raise ValueError(f"Column '{self._run_id_column}' not found in dataframe")

        train_dfs = []
        test_dfs = []

        for run_id in df[self._run_id_column].unique():
            run_data = df[df[self._run_id_column] == run_id].copy()

            # Sort by timestamp to maintain temporal order
            if 'timestamp' in run_data.columns:
                run_data = run_data.sort_values('timestamp').reset_index(drop=True)

            # Use stratified split for each CSV/run if possible (if stratify column exists & has >1 class)
            if 'health_status' in run_data.columns and len(run_data['health_status'].unique()) > 1:
                stratify_labels = run_data['health_status']
            else:
                stratify_labels = None

            train_data, test_data = train_test_split(
                run_data,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels
            )

            # Sort back by timestamp after splitting to preserve temporal order
            if 'timestamp' in run_data.columns:
                train_data = train_data.sort_values('timestamp').reset_index(drop=True)
                test_data = test_data.sort_values('timestamp').reset_index(drop=True)

            train_dfs.append(train_data)
            test_dfs.append(test_data)

        # Combine all splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        return train_df, test_df


class StratifiedSplitter(IDataSplitter):
    """Splits data using stratified sampling on target column."""

    def __init__(self, stratify_column: str = 'health_status'):
        self._stratify_column = stratify_column

    def split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[self._stratify_column]
        )


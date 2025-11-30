"""Data splitting implementations."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.interfaces import IDataSplitter


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


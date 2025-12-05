"""Filters for excluding/including specific health conditions."""
import pandas as pd
from src.core.interfaces import IDataFilter


class ExcludeNormalFilter(IDataFilter):
    """Excludes rows with Normal health status."""
    
    def __init__(self, status_column: str = 'health_status'):
        self._status_column = status_column
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self._status_column] != 'Normal'].copy()


class IncludeOnlyNormalFilter(IDataFilter):
    """Includes only rows with Normal health status."""
    
    def __init__(self, status_column: str = 'health_status'):
        self._status_column = status_column
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self._status_column] == 'Normal'].copy()


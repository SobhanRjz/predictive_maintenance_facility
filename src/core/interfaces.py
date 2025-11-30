"""Core interfaces for preprocessing components."""
from abc import ABC, abstractmethod
from typing import Protocol
import pandas as pd


class IDataLoader(ABC):
    """Interface for loading motor sensor data."""
    
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load data from specified path."""
        pass


class IDataFilter(ABC):
    """Interface for filtering data by condition."""
    
    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe based on health status."""
        pass


class IDataSplitter(ABC):
    """Interface for splitting data into train/test sets."""
    
    @abstractmethod
    def split(self, df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test sets."""
        pass


class IDataExporter(ABC):
    """Interface for exporting processed data."""
    
    @abstractmethod
    def export(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
        """Export train and test datasets."""
        pass


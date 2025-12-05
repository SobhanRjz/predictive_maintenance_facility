"""Interfaces for backend components."""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from influxdb_client import InfluxDBClient, Point

from datetime import datetime


class IInfluxDBClient(ABC):
    """Interface for InfluxDB client operations."""

    @abstractmethod
    def create_client(self) -> InfluxDBClient:
        """Create and return an InfluxDB client."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the InfluxDB client."""
        pass


class ICSVLoader(ABC):
    """Interface for CSV loading operations."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load CSV into a pandas DataFrame with parsed timestamps."""
        pass


class IDataConverter(ABC):
    """Interface for converting data to InfluxDB points."""

    @abstractmethod
    def row_to_point(self, row: pd.Series, measurement_name: str) -> Tuple[Point, datetime]:
        """Convert one CSV row into an InfluxDB Point."""
        pass


class IDataStreamer(ABC):
    """Interface for streaming data to InfluxDB."""

    @abstractmethod
    def stream(self, df: pd.DataFrame) -> None:
        """Stream dataframe to InfluxDB."""
        pass
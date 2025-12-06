"""Centralized configuration for CSV generator."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class InfluxDBConfig:
    """Configuration for InfluxDB connection."""
    url: str
    token: str
    org: str
    bucket: str


@dataclass
class CSVConfig:
    """Configuration for CSV processing."""
    path: str
    measurement_name: str
    status_map: Dict[str, int]
    sleep_seconds: float
    loop_forever: bool
    is_csv_status_enabled: bool
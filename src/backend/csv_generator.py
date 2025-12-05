"""CSV generator for streaming data to InfluxDB."""

import logging
import time
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions

from config.csv_config import InfluxDBConfig, CSVConfig
from config.logger_config import LoggerConfig
from interfaces import IInfluxDBClient, ICSVLoader, IDataConverter, IDataStreamer


class InfluxDBClientManager(IInfluxDBClient):
    """Manages InfluxDB client lifecycle."""

    def __init__(self, config: InfluxDBConfig):
        self.config = config
        self._client: InfluxDBClient = None

    def create_client(self) -> InfluxDBClient:
        """Create and return an InfluxDB client."""
        if self._client is None:
            self._client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
            )
        return self._client

    def close(self) -> None:
        """Close the InfluxDB client."""
        if self._client:
            self._client.close()
            self._client = None


class CSVLoader(ICSVLoader):
    """Handles CSV file loading and validation."""

    def load(self, path: str) -> pd.DataFrame:
        """Load CSV into a pandas DataFrame with parsed timestamps."""
        df = pd.read_csv(path)

        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


class DataConverter(IDataConverter):
    """Converts CSV rows to InfluxDB points."""

    def __init__(self, status_map: dict):
        self.status_map = status_map

    def row_to_point(self, row: pd.Series, measurement_name: str) -> Tuple[Point, datetime]:
        """Convert one CSV row into an InfluxDB Point."""
        now = datetime.now(timezone.utc)
        p = Point(measurement_name).time(now)

        # Handle health_status specially
        if "health_status" in row.index:
            status_str = str(row["health_status"])
            code = self.status_map.get(status_str, -1)

            # Store as tag (string)
            p = p.tag("health_status", status_str)

            # Store as numeric field
            p = p.field("health_status_code", int(code))

        # Add all other numeric fields
        for col in row.index:
            if col in ("timestamp", "health_status"):
                continue

            value = row[col]
            if pd.isna(value):
                continue

            try:
                p = p.field(col, float(value))
            except (ValueError, TypeError):
                # Ignore non-numeric
                pass

        return p, now


class DataStreamer(IDataStreamer):
    """Streams CSV data to InfluxDB."""

    def __init__(self, client_manager: IInfluxDBClient, converter: IDataConverter,
                 influx_config: InfluxDBConfig, csv_config: CSVConfig,
                 logger: logging.Logger):
        self.client_manager = client_manager
        self.converter = converter
        self.influx_config = influx_config
        self.csv_config = csv_config
        self.logger = logger

    def stream(self, df: pd.DataFrame) -> None:
        """Stream dataframe to InfluxDB."""
        client = self.client_manager.create_client()
        write_api = client.write_api(write_options=WriteOptions(batch_size=1))

        try:
            while True:
                for index, row in df.iterrows():
                    point, now = self.converter.row_to_point(row, self.csv_config.measurement_name)
                    write_api.write(bucket=self.influx_config.bucket, org=self.influx_config.org, record=point)
                    self.logger.info(f"Wrote point for row {index} at {now.isoformat()}")

                    time.sleep(self.csv_config.sleep_seconds)

                if not self.csv_config.loop_forever:
                    self.logger.info("Finished streaming CSV once. Exiting.")
                    break
                else:
                    self.logger.info("Reached end of CSV, starting again from top...")

        finally:
            self.client_manager.close()


class CSVGeneratorOrchestrator:
    """Orchestrates the CSV generation process."""

    def __init__(self, influx_config: InfluxDBConfig, csv_config: CSVConfig):
        self.influx_config = influx_config
        self.csv_config = csv_config

        # Initialize logger
        self.logger = LoggerConfig.setup_logger()

        # Initialize components with dependency injection
        self.client_manager = InfluxDBClientManager(influx_config)
        self.loader = CSVLoader()
        self.converter = DataConverter(csv_config.status_map)
        self.streamer = DataStreamer(self.client_manager, self.converter, influx_config, csv_config, self.logger)

    def execute(self) -> None:
        """Execute the complete CSV generation workflow."""
        df = self.loader.load(self.csv_config.path)
        self.streamer.stream(df)


# Default configuration instance
def create_default_config() -> Tuple[InfluxDBConfig, CSVConfig]:
    """Create default configuration instances."""
    influx_config = InfluxDBConfig(
        url="http://localhost:8086",
        token="NxHBp_AI8xLI5gGdvFvXN1Ye-fLB73EWGa8i0Q14FM7LwFJwNpULHuxdZhi0GWKacLc-iii7LCGOiFvXLxLPgw==",
        org="csv_org",
        bucket="f_sensor"
    )

    csv_config = CSVConfig(
        path=r"datasets\datasets_renamed\normal_3months_30sec_interval-1.csv",
        measurement_name="sensor_measurements",
        status_map={
            "Normal": 0,
            "Warning": 1,
            "Failure": 2,
            "Error": 2,
        },
        sleep_seconds=2.0,
        loop_forever=True
    )

    return influx_config, csv_config


if __name__ == "__main__":
    influx_config, csv_config = create_default_config()
    orchestrator = CSVGeneratorOrchestrator(influx_config, csv_config)
    orchestrator.execute()
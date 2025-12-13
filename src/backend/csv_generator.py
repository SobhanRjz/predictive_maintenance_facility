"""CSV generator for streaming data to InfluxDB."""

import logging
import time
from datetime import datetime, timezone
from typing import Tuple, List

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WriteOptions

from src.backend.config import InfluxDBConfig, CSVConfig
from src.backend.config import LoggerConfig
from src.backend.interfaces import IInfluxDBClient, ICSVLoader, IDataConverter, IDataStreamer

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__)))))
# Import ML components
from src.ML.inference.hierarchical_predictor import HierarchicalPredictor
from src.ML.features.time_domain_features import TimeDomainFeatureExtractor


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
        df.dropna(inplace=True)
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

        # Handle fault_type tag
        if "fault_type" in row.index and pd.notna(row["fault_type"]):
            p = p.tag("fault_type", str(row["fault_type"]))

        # Add all other numeric fields
        for col in row.index:
            if col in ("timestamp", "health_status", "fault_type"):
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
        self._is_csv_status_enabled = csv_config.is_csv_status_enabled

        # Initialize ML inference
        try:
            self.feature_extractor = TimeDomainFeatureExtractor(
                window_size="10min",
                timestamp_col="timestamp",
                target_col="health_status"
            )
            self.hierarchical_predictor = HierarchicalPredictor(
                layer1_model_path="models/layer1_anomaly_detection.pkl",
                layer2_warning_model_path="models/layer2_warning_classifier.pkl",
                layer2_failure_model_path="models/layer2_failure_classifier.pkl"
            )
            self.logger.info("Hierarchical predictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize hierarchical predictor: {e}")
            self.feature_extractor = None
            self.hierarchical_predictor = None

    def stream(self, df: pd.DataFrame) -> None:
        """Stream dataframe to InfluxDB."""
        client = self.client_manager.create_client()
        write_api = client.write_api(write_options=WriteOptions(batch_size=1))
        status_str = "Normal"
        fault_type_str = "none"
        try:
            while True:
                batch_rows: List[pd.Series] = []
                batch_indices: List[int] = []

                for index, row in df.iterrows():
                    # Collect row data for batch processing
                    # Check that none of the values in the row are None or empty string
                    # Check for None, empty strings, or any NaN in the row
                    if any(
                        v is None or (isinstance(v, str) and v.strip() == "")
                        for v in row
                    ) or row.isna().any():
                        self.logger.warning(f"Skipping row {index} due to None, empty string, or NaN values: {row.to_dict()}")
                    else:
                        batch_rows.append(row.copy())
                        batch_indices.append(index)


                    if not self._is_csv_status_enabled:
                        # Process batch when we have 5 rows
                        if len(batch_rows) == 5:
                            try:
                                status_str, fault_type_str = self._process_ml_batch(batch_rows, batch_indices)

                                # normalize fault_type for normal
                                if status_str == "Normal":
                                    fault_type_str = "none"

                                batch_rows.clear()
                                batch_indices.clear()
                            except Exception as e:
                                self.logger.error(f"Error processing ML batch: {e}")


                    # ALWAYS write these two columns for every row
                    row["health_status"] = status_str
                    row["fault_type"] = fault_type_str

                    # Write to InfluxDB
                    point, now = self.converter.row_to_point(row, self.csv_config.measurement_name)
                    write_api.write(bucket=self.influx_config.bucket, org=self.influx_config.org, record=point)
                    self.logger.info(f"Wrote point for row {index} at {now.isoformat()}")



                    time.sleep(self.csv_config.sleep_seconds)

                status_str = "Normal"
                fault_type_str = "none"

                if not self.csv_config.loop_forever:
                    self.logger.info("Finished streaming CSV once. Exiting.")
                    break
                else:
                    self.logger.info("Reached end of CSV, starting again from top...")

        finally:
            self.client_manager.close()

    def _process_ml_batch(self, rows: List[pd.Series], indices: List[int]) -> Tuple[str, str]:
        """Process a batch through hierarchical predictor."""
        if not self.hierarchical_predictor or not self.feature_extractor:
            self.logger.warning("Hierarchical predictor not available")
            return "Normal", "none"

        try:
            # Create DataFrame from batch rows
            batch_df = pd.DataFrame(rows)

            # Extract features
            features_df = self.feature_extractor.extract(batch_df)

            # Run hierarchical predictions
            results_df = self.hierarchical_predictor.predict(features_df)

            # Get first prediction result
            result = results_df.iloc[0]
            status_str = result['health_status']
            fault_type = result.get('failure_type') or result.get('warning_type')

            return status_str, fault_type

        except Exception as e:
            self.logger.error(f"Error processing ML batch: {e}")
            return "Normal", "none"


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
        path=r"C:\Users\sobha\Desktop\TestInfluxDB\ML_GasOil\datasets\datasets_renamed\TestForModel_Layer1.csv",
        measurement_name="sensor_measurements",
        status_map={
            "Normal": 0,
            "Warning": 1,
            "Failure": 2,
            "Error": 2,
        },
        sleep_seconds=2.0,
        loop_forever=True,
        is_csv_status_enabled=False
    )

    return influx_config, csv_config


if __name__ == "__main__":
    influx_config, csv_config = create_default_config()
    orchestrator = CSVGeneratorOrchestrator(influx_config, csv_config)
    orchestrator.execute()
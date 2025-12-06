"""Main entry point for preprocessing pipeline."""
import logging
from src.backend.csv_generator import CSVGeneratorOrchestrator, create_default_config
from src.backend.config import InfluxDBConfig, CSVConfig, LoggerConfig

# Configure logging
logger = LoggerConfig.setup_logger()

def main():
	logger.info("Starting CSV temperature creation pipeline")
	try:
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

		orchestrator = CSVGeneratorOrchestrator(influx_config, csv_config)
		orchestrator.execute()
		logger.info("CSV temperature creation pipeline completed successfully")
	except Exception as e:
		logger.error(f"Error in CSV temperature creation pipeline: {e}")
		raise


if __name__ == '__main__':
	main()


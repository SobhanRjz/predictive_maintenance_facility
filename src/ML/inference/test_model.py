"""Model testing utility for gas/oil equipment health status prediction."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
from src.ML.features.time_domain_features import TimeDomainFeatureExtractor
from src.ML.models.xgboost_model import XGBoostModel
from src.ML.loaders.data_loader import MultiFileLoader
from src.ML.filters.condition_filter import IncludeOnlyNormalFilter, ExcludeNormalFilter
from src.ML.validators.data_validator import DataValidator
from src.ML.loaders.data_loader import DataResampler, CSVLoader, ExcelLoader

class ModelTester:
    """Tests trained XGBoost model with raw sensor data using full preprocessing pipeline."""

    def __init__(
        self,
        model_path: str = "models/xgboost_model.pkl",
        window_size: str = "10min",
        timestamp_col: str = "timestamp",
        target_col: str = "health_status",
        start_run_id: int = 1
    ):
        """
        Args:
            model_path: Path to saved model file
            window_size: Time window for feature extraction
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            start_run_id: Starting run ID for data loading
        """
        self._model_path = Path(model_path)
        self._start_run_id = start_run_id

        resampler = DataResampler(freq='1T', agg_method='mean')

        csv_loader = CSVLoader()
        excel_loader = ExcelLoader()
        # Initialize preprocessing components (same as preprocessing orchestrator)
        self._loader = MultiFileLoader(csv_loader, excel_loader, resampler)
        self._normal_filter = IncludeOnlyNormalFilter()
        self._abnormal_filter = ExcludeNormalFilter()
        self._validator = DataValidator()
        self._feature_extractor = TimeDomainFeatureExtractor(
            window_size=window_size,
            timestamp_col=timestamp_col,
            target_col=target_col
        )

        # Initialize model
        self._model = XGBoostModel()
        self._load_model()

    def preprocess_data(self, data_paths: Union[str, List[str]], condition_type: str = "unknown") -> pd.DataFrame:
        """
        Preprocess raw sensor data files using the same pipeline as training.

        Args:
            data_paths: Path(s) to CSV file(s) with raw sensor data
            condition_type: Type of condition data ("normal", "abnormal", or "unknown")

        Returns:
            DataFrame with extracted features ready for prediction
        """
        # Handle single path vs multiple paths
        if isinstance(data_paths, str):
            paths_list = [data_paths]
        else:
            paths_list = data_paths

        print(f"Loading {len(paths_list)} data file(s)...")
        df = self._loader.load_multiple(paths_list, start_run_id=self._start_run_id)

        # Apply appropriate filter based on condition type
        if condition_type == "normal":
            df = self._normal_filter.filter(df)
            print(f"Applied normal condition filter: {len(df)} rows")
        elif condition_type == "abnormal":
            df = self._abnormal_filter.filter(df)
            print(f"Applied abnormal condition filter: {len(df)} rows")
        else:
            print(f"Loaded data without filtering: {len(df)} rows")

        # Validate data quality
        is_valid, validation_report = self._validator.validate(df)
        if not is_valid:
            print("Warning: Data validation detected issues:")
            for issue in validation_report.get('issues', []):
                print(f"  - {issue}")

        # Extract features
        print("Extracting features...")
        features_df = self._feature_extractor.extract(df)
        print(f"Features extracted: {features_df.shape}")

        return features_df

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self._model.load(str(self._model_path))

    def predict_from_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions from raw sensor data (includes preprocessing).

        Args:
            df: DataFrame with columns: timestamp, sensor readings, health_status (optional)

        Returns:
            DataFrame with predictions and probabilities
        """
        # Preprocess the data (extract features)
        features_df = self._feature_extractor.extract(df)

        # Make predictions using the processed features
        return self.predict_from_features(features_df)

    def predict_from_files(self, data_paths: Union[str, List[str]], condition_type: str = "unknown") -> pd.DataFrame:
        """
        Make predictions from raw sensor data files (includes full preprocessing pipeline).

        Args:
            data_paths: Path(s) to CSV file(s) with raw sensor data
            condition_type: Type of condition data ("normal", "abnormal", or "unknown")

        Returns:
            DataFrame with predictions and probabilities
        """
        # Preprocess the data files
        features_df = self.preprocess_data(data_paths, condition_type)

        # Make predictions using the processed features
        return self.predict_from_features(features_df)

    def predict_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions from preprocessed feature data.

        Args:
            features_df: DataFrame with extracted features

        Returns:
            DataFrame with predictions and probabilities
        """
        # Prepare features for prediction using stored feature names
        feature_names = self._model.get_feature_names()
        if feature_names is None:
            raise RuntimeError("Model was not trained with feature names. Retrain the model to enable inference.")
        X = features_df[feature_names].values

        # Make predictions
        predictions = self._model.predict(X)
        probabilities = self._model.predict_proba(X)

        # Get class names
        class_names = self._model._label_encoder.classes_

        # Create results DataFrame
        results = features_df[['timestamp']].copy()
        results['predicted_health_status'] = predictions

        # Add probability columns
        for i, class_name in enumerate(class_names):
            results[f'prob_{class_name}'] = probabilities[:, i]

        return results

    def predict_single_window(self, sensor_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for single time window of sensor readings.

        Args:
            sensor_data: Dictionary with sensor column names as keys and values as values

        Returns:
            Dictionary with prediction and probabilities
        """
        # Create a DataFrame with single row
        df = pd.DataFrame([sensor_data])
        df['timestamp'] = pd.Timestamp.now()

        # Make prediction (includes feature extraction for single row)
        results = self.predict_from_raw_data(df)

        # Convert to dictionary
        result_dict = results.iloc[0].to_dict()

        # Remove timestamp from output
        result_dict.pop('timestamp', None)

        return result_dict

    @staticmethod
    def test_with_preprocessing_example():
        """
        Example of how to use the ModelTester with full preprocessing pipeline.

        This demonstrates the recommended approach for testing with raw sensor data files.
        """
        # Initialize tester
        tester = ModelTester()

        # Example 1: Test with raw sensor data files
        # test_data_paths = ["data/test/normal_condition.csv", "data/test/abnormal_condition.csv"]
        # results = tester.predict_from_files(test_data_paths, condition_type="unknown")

        # Example 2: Test with DataFrame of raw sensor data
        # raw_data = ModelTester.create_sample_data(50)
        # results = tester.predict_from_raw_data(raw_data)

        # Example 3: Test single sensor reading window
        # sensor_reading = {
        #     'accelerometer_g': 0.3,
        #     'vibration_velocity_mm_s': 3.2,
        #     'shaft_displacement_um': 40.0,
        #     # ... other sensor readings
        # }
        # result = tester.predict_single_window(sensor_reading)

        print("ModelTester now includes full preprocessing pipeline!")
        print("Use predict_from_files() for raw data files or predict_from_raw_data() for DataFrames")

    @staticmethod
    def create_sample_data(n_rows: int = 100) -> pd.DataFrame:
        """
        Create sample sensor data for testing.

        Args:
            n_rows: Number of sample rows to generate

        Returns:
            DataFrame with sample sensor readings
        """
        np.random.seed(42)

        # Generate timestamps
        base_time = pd.Timestamp('2024-01-01')
        timestamps = [base_time + pd.Timedelta(minutes=i*10) for i in range(n_rows)]

        # Sensor column names based on the dataset
        sensor_cols = [
            'accelerometer_g', 'vibration_velocity_mm_s', 'shaft_displacement_um',
            'bearing_temp_c', 'oil_temp_c', 'casing_temp_c',
            'inlet_fluid_temp_c', 'outlet_fluid_temp_c',
            'inlet_pressure_bar', 'outlet_pressure_bar',
            'flow_rate_m3_h', 'motor_current_a', 'supply_voltage_v',
            'power_consumption_kw', 'sound_intensity_db'
        ]

        # Generate realistic ranges for each sensor
        sensor_ranges = {
            'accelerometer_g': (0.1, 0.5),
            'vibration_velocity_mm_s': (2.5, 4.0),
            'shaft_displacement_um': (30, 50),
            'bearing_temp_c': (60, 70),
            'oil_temp_c': (50, 60),
            'casing_temp_c': (55, 65),
            'inlet_fluid_temp_c': (40, 50),
            'outlet_fluid_temp_c': (50, 60),
            'inlet_pressure_bar': (2.0, 3.0),
            'outlet_pressure_bar': (6.0, 7.0),
            'flow_rate_m3_h': (85, 95),
            'motor_current_a': (35, 45),
            'supply_voltage_v': (395, 410),
            'power_consumption_kw': (18, 22),
            'sound_intensity_db': (70, 80)
        }

        # Generate sample data
        data = {}
        for col in sensor_cols:
            min_val, max_val = sensor_ranges[col]
            data[col] = np.random.uniform(min_val, max_val, n_rows)

        data['timestamp'] = timestamps

        return pd.DataFrame(data)

    @staticmethod
    def create_custom_data(sensor_readings: Union[Dict[str, float], List[Dict[str, float]]],
                          timestamps: Union[str, pd.Timestamp, List[Union[str, pd.Timestamp]]] = None) -> pd.DataFrame:
        """
        Create DataFrame from your custom sensor readings.

        Args:
            sensor_readings: Single dict or list of dicts with sensor readings
            timestamps: Single timestamp or list of timestamps (auto-generated if None)

        Returns:
            DataFrame ready for prediction

        Example:
            # Single row
            data = {
                'accelerometer_g': 0.3,
                'vibration_velocity_mm_s': 3.2,
                'shaft_displacement_um': 40.0,
                'bearing_temp_c': 65.0,
                'oil_temp_c': 55.0,
                'casing_temp_c': 60.0,
                'inlet_fluid_temp_c': 45.0,
                'outlet_fluid_temp_c': 55.0,
                'inlet_pressure_bar': 2.5,
                'outlet_pressure_bar': 6.5,
                'flow_rate_m3_h': 90.0,
                'motor_current_a': 40.0,
                'supply_voltage_v': 400.0,
                'power_consumption_kw': 20.0,
                'sound_intensity_db': 75.0
            }
            df = ModelTester.create_custom_data(data)

            # Multiple rows
            rows = [data1, data2, data3]
            df = ModelTester.create_custom_data(rows)
        """
        # Required sensor columns
        required_cols = [
            'accelerometer_g', 'vibration_velocity_mm_s', 'shaft_displacement_um',
            'bearing_temp_c', 'oil_temp_c', 'casing_temp_c',
            'inlet_fluid_temp_c', 'outlet_fluid_temp_c',
            'inlet_pressure_bar', 'outlet_pressure_bar',
            'flow_rate_m3_h', 'motor_current_a', 'supply_voltage_v',
            'power_consumption_kw', 'sound_intensity_db'
        ]

        # Handle single dict vs list of dicts
        if isinstance(sensor_readings, dict):
            readings_list = [sensor_readings]
        else:
            readings_list = sensor_readings

        # Validate required columns
        for i, reading in enumerate(readings_list):
            missing_cols = set(required_cols) - set(reading.keys())
            if missing_cols:
                raise ValueError(f"Row {i} missing required columns: {missing_cols}")

        # Handle timestamps
        if timestamps is None:
            # Auto-generate timestamps
            base_time = pd.Timestamp.now()
            timestamps_list = [base_time + pd.Timedelta(minutes=i*10) for i in range(len(readings_list))]
        elif isinstance(timestamps, (str, pd.Timestamp)):
            timestamps_list = [pd.Timestamp(timestamps)] * len(readings_list)
        else:
            timestamps_list = [pd.Timestamp(t) for t in timestamps]

        if len(timestamps_list) != len(readings_list):
            raise ValueError("Number of timestamps must match number of sensor readings")

        # Create DataFrame
        df = pd.DataFrame(readings_list)
        df['timestamp'] = timestamps_list

        # Reorder columns to match expected format
        cols = ['timestamp'] + required_cols
        df = df[cols]

        return df

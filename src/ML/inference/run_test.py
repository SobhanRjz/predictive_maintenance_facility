"""Simple script to run model tests."""
from test_model import ModelTester
import pandas as pd

def test_with_sample_data():
    """Test model with generated sample data."""
    print("Testing with sample data...")

    tester = ModelTester()

    # Generate sample data
    sample_df = tester.create_sample_data(n_rows=50)
    print(f"Generated {len(sample_df)} sample rows")

    # Make predictions
    results = tester.predict_from_raw_data(sample_df)

    print("\nPrediction results (first 5 rows):")
    print(results.head())

    # Test single window prediction
    sample_row = sample_df.iloc[0].to_dict()
    single_result = tester.predict_single_window(sample_row)

    print(f"\nSingle window prediction: {single_result['predicted_health_status']}")

def test_with_custom_rows():
    """Test model with your custom sensor readings."""
    print("Testing with custom sensor data...")

    tester = ModelTester()

    # Example: Single row of sensor readings
    custom_data = {
        'accelerometer_g': 0.35,
        'vibration_velocity_mm_s': 3.8,
        'shaft_displacement_um': 42.5,
        'bearing_temp_c': 68.2,
        'oil_temp_c': 57.1,
        'casing_temp_c': 62.8,
        'inlet_fluid_temp_c': 46.5,
        'outlet_fluid_temp_c': 56.3,
        'inlet_pressure_bar': 2.7,
        'outlet_pressure_bar': 6.8,
        'flow_rate_m3_h': 92.1,
        'motor_current_a': 41.5,
        'supply_voltage_v': 402.3,
        'power_consumption_kw': 20.8,
        'sound_intensity_db': 77.2
    }

    # Create DataFrame from custom data
    df = tester.create_custom_data(custom_data)
    print("Created DataFrame with custom sensor readings")

    # Make prediction
    result = tester.predict_single_window(custom_data)
    print(f"Prediction result: {result}")

    return result

def test_with_multiple_custom_rows():
    """Test model with multiple custom sensor readings."""
    print("Testing with multiple custom sensor readings...")

    tester = ModelTester()

    # Example: Multiple rows of sensor readings
    custom_rows = [
        {
            'accelerometer_g': 0.32, 'vibration_velocity_mm_s': 3.5, 'shaft_displacement_um': 38.2,
            'bearing_temp_c': 65.1, 'oil_temp_c': 54.8, 'casing_temp_c': 59.3,
            'inlet_fluid_temp_c': 44.2, 'outlet_fluid_temp_c': 53.9,
            'inlet_pressure_bar': 2.4, 'outlet_pressure_bar': 6.3,
            'flow_rate_m3_h': 89.5, 'motor_current_a': 39.2,
            'supply_voltage_v': 398.7, 'power_consumption_kw': 19.6,
            'sound_intensity_db': 74.1
        },
        {
            'accelerometer_g': 0.41, 'vibration_velocity_mm_s': 4.2, 'shaft_displacement_um': 45.8,
            'bearing_temp_c': 71.5, 'oil_temp_c': 59.3, 'casing_temp_c': 64.7,
            'inlet_fluid_temp_c': 47.8, 'outlet_fluid_temp_c': 58.2,
            'inlet_pressure_bar': 2.9, 'outlet_pressure_bar': 7.1,
            'flow_rate_m3_h': 94.3, 'motor_current_a': 43.8,
            'supply_voltage_v': 405.2, 'power_consumption_kw': 21.4,
            'sound_intensity_db': 79.8
        }
    ]

    # Create DataFrame from multiple custom rows
    df = tester.create_custom_data(custom_rows)
    print(f"Created DataFrame with {len(df)} custom sensor readings")

    # Make predictions
    results = tester.predict_from_raw_data(df)
    print("Prediction results:")
    print(results[['timestamp', 'predicted_health_status']])

    return results

def test_with_sensor_data(sensor_data_list: list, window_size: int = 10):
    """Test model with sensor data provided as list of lists."""
    print("Testing with sensor data list...")

    tester = ModelTester()

    # Sensor column names (from your config)
    sensor_columns = [
        'accelerometer_g', 'vibration_velocity_mm_s', 'shaft_displacement_um',
        'bearing_temp_c', 'oil_temp_c', 'casing_temp_c',
        'inlet_fluid_temp_c', 'outlet_fluid_temp_c',
        'inlet_pressure_bar', 'outlet_pressure_bar',
        'flow_rate_m3_h', 'motor_current_a',
        'supply_voltage_v', 'power_consumption_kw',
        'sound_intensity_db'
    ]

    # Convert list of lists to list of dictionaries
    rows = []
    for sensor_values in sensor_data_list:
        if len(sensor_values) == len(sensor_columns):
            row_dict = dict(zip(sensor_columns, sensor_values))
            rows.append(row_dict)

    print(f"Original data has {len(rows)} rows")

    # If we don't have enough data for feature extraction, duplicate to fill the window
    if len(rows) < window_size:
        print(f"Duplicating data to create {window_size} rows for feature extraction...")
        # Duplicate the existing rows to fill the window
        duplicated_rows = []
        repetitions = (window_size // len(rows)) + 1
        for _ in range(repetitions):
            duplicated_rows.extend(rows)
        rows = duplicated_rows[:window_size]  # Take exactly window_size rows
        print(f"Created {len(rows)} rows by duplication")

    # Create DataFrame from parsed rows
    df = tester.create_custom_data(rows)
    print(f"Created DataFrame with {len(df)} sensor readings")

    # Make predictions
    results = tester.predict_from_raw_data(df)

    print("Prediction results:")
    print(results[['timestamp', 'predicted_health_status']])

    return results

def test_with_csv_rows(csv_path: str, start_row: int = None, end_row: int = None, n_rows: int = None):
    """
    Test model with rows from a CSV file.

    Args:
        csv_path: Path to CSV file
        start_row: Starting row index (0-based, inclusive)
        end_row: Ending row index (0-based, exclusive)
        n_rows: Number of rows to use (alternative to start_row/end_row)
    """
    tester = ModelTester()
    # results0 = tester.predict_from_files(csv_path, condition_type="")
    # print("Prediction results:")
    # print(results0[['timestamp', 'predicted_health_status']])
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Select rows based on parameters
    if start_row is not None and end_row is not None:
        # Use specific row range
        selected_rows = df.iloc[start_row:end_row]
        print(f"Using rows {start_row} to {end_row-1} ({len(selected_rows)} rows)")
    elif n_rows is not None:
        # Use last N rows (backward compatibility)
        selected_rows = df.tail(n_rows)
        print(f"Using last {len(selected_rows)} rows")
    else:
        # Default to last 10 rows
        selected_rows = df.tail(10)
        print(f"Using last {len(selected_rows)} rows (default)")

    # Make predictions
    results = tester.predict_from_raw_data(selected_rows)

    print("Prediction results:")
    print(results[['timestamp', 'predicted_health_status']])

    return results

def test_with_your_data(csv_path: str):
    """
    Test model with your own CSV data.

    Expected CSV format:
    timestamp,accelerometer_g,vibration_velocity_mm_s,shaft_displacement_um,
    bearing_temp_c,oil_temp_c,casing_temp_c,inlet_fluid_temp_c,outlet_fluid_temp_c,
    inlet_pressure_bar,outlet_pressure_bar,flow_rate_m3_h,motor_current_a,
    supply_voltage_v,power_consumption_kw,sound_intensity_db
    """
    print(f"Testing with your data from {csv_path}...")

    tester = ModelTester()

    # Load your data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Make predictions
    results = tester.predict_from_raw_data(df)

    # Save results
    output_path = csv_path.replace('.csv', '_predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return results

if __name__ == '__main__':
    # Test with rows from failure_1_bearing_fault.csv
    csv_path = r"C:\Users\sobha\Desktop\TestInfluxDB\ML_GasOil\datasets\datasets_renamed\failure_1_bearing_fault.csv"

    
    # Example 1: Last 10 rows (equivalent to previous behavior)
    print("Example 1: Last 10 rows")
    test_with_csv_rows(csv_path, start_row=126721, end_row=126761, n_rows=10)

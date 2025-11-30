"""Configuration for preprocessing pipeline."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and preprocessing parameters."""
    
    datasets_dir: str = 'datasets\\datasets_renamed'
    output_dir: str = 'datasets\\processed_data'
    test_size: float = 0.2
    random_state: int = 42
    use_timeseries_split: bool = True
    use_stratification: bool = True
    window_size: str = '10T'  # 10 minutes for 1-minute interval data

    # Resampling configuration (30sec -> 1min) all files should be same sampling rate
    resample_freq: str = '1min'  # 1 minute
    resample_method: str = 'mean'  # Use mean for sensor signals

    # Renamed according to file_renamer.py (56-88)
    normal_files: list[str] = field(default_factory=lambda: [
        'normal_3months_30sec_interval-1.csv'
    ])

    failure_files: list[str] = field(default_factory=lambda: [
        'failure_1_bearing_fault.csv',
        # 'failure_2_shaft_misalignment.csv',
        # 'failure_3_rotor_imbalance.csv',
        # 'failure_4_cavitation.csv',
        # 'failure_5_pipe_blockage.csv',
        # 'failure_6_motor_overload.csv',
        # 'failure_7_voltage_drop.csv',
        # 'failure_8_seal_failure.csv',
        # 'failure_9_impeller_wear.csv'
    ])

    warning_files: list[str] = field(default_factory=lambda: [
        'warning_1_radial_vibration_increase.csv',
        # 'warning_2_axial_vibration_increase.csv',
        # 'warning_3_bearing_temp_increase.csv',
        # 'warning_4_oil_temp_increase.csv',
        # 'warning_5_casing_temp_increase.csv',
        # 'warning_6_suction_pressure_drop.csv',
        # 'warning_7_discharge_pressure_drop.csv',
        # 'warning_8_flow_rate_decrease.csv',
        # 'warning_9_power_increase.csv',
        # 'warning_10_current_increase.csv',
        # 'warning_11_voltage_drop.csv',
        # 'warning_12_acoustic_noise_increase.csv',
        # 'warning_13_outlet_fluid_temp_increase.csv',
        # 'warning_14_flow_pressure_power_fluctuation.csv',
        # 'warning_15_bearing_vibration_temp_increase.csv'
    ])
    
    def get_normal_paths(self) -> list[str]:
        """Get full paths to normal condition files."""
        return [str(Path(self.datasets_dir) / f) for f in self.normal_files]
    
    def get_abnormal_paths(self) -> list[str]:
        """Get full paths to failure and warning files."""
        all_files = self.failure_files + self.warning_files
        return [str(Path(self.datasets_dir) / f) for f in all_files]

    def get_converted_paths(self, converted_dir: str = 'datasets_converted') -> tuple[list[str], list[str]]:
        """
        Get paths to converted CSV files instead of original files.

        Returns:
            Tuple of (normal_paths, abnormal_paths) pointing to converted CSV files
        """
        # Convert normal files to CSV names
        normal_csv_files = []
        for normal_file in self.normal_files:
            if normal_file.endswith('.csv'):
                csv_name = self._get_converted_filename(normal_file)
                normal_csv_files.append(csv_name)

        # Convert abnormal files to CSV names
        abnormal_csv_files = []
        all_abnormal = self.failure_files + self.warning_files
        for abnormal_file in all_abnormal:
            csv_name = self._get_converted_filename(abnormal_file)
            abnormal_csv_files.append(csv_name)

        normal_paths = [str(Path(converted_dir) / f) for f in normal_csv_files]
        abnormal_paths = [str(Path(converted_dir) / f) for f in abnormal_csv_files]

        return normal_paths, abnormal_paths

    def _get_converted_filename(self, original_filename: str) -> str:
        """Convert original filename to its CSV equivalent."""
        renamer = FileRenamer()
        if original_filename in renamer.FILENAME_MAP:
            return renamer.FILENAME_MAP[original_filename]
        else:
            # Fallback: just change extension to .csv
            return original_filename.replace('.xlsx', '.csv').replace('.xls', '.csv')


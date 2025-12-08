"""Label extractor for hierarchical classification targets."""
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LabelExtractor:
    """Extracts fine-grained labels from filenames for hierarchical models."""
    
    @staticmethod
    def extract_warning_type(filename: str) -> str:
        """Extract warning type from filename."""
        name = Path(filename).stem
        
        if 'radial_vibration_increase' in name:
            return 'radial_vibration_increase'
        elif 'axial_vibration_increase' in name:
            return 'axial_vibration_increase'
        elif 'bearing_temp_increase' in name:
            return 'bearing_temp_increase'
        elif 'oil_temp_increase' in name:
            return 'oil_temp_increase'
        elif 'casing_temp_increase' in name:
            return 'casing_temp_increase'
        elif 'suction_pressure_drop' in name:
            return 'suction_pressure_drop'
        elif 'discharge_pressure_drop' in name:
            return 'discharge_pressure_drop'
        elif 'flow_rate_decrease' in name:
            return 'flow_rate_decrease'
        elif 'power_increase' in name:
            return 'power_increase'
        elif 'current_increase' in name:
            return 'current_increase'
        elif 'voltage_drop' in name:
            return 'voltage_drop'
        elif 'acoustic_noise_increase' in name:
            return 'acoustic_noise_increase'
        elif 'outlet_fluid_temp_increase' in name:
            return 'outlet_fluid_temp_increase'
        elif 'flow_pressure_power_fluctuation' in name:
            return 'flow_pressure_power_fluctuation'
        elif 'bearing_vibration_temp_increase' in name:
            return 'bearing_vibration_temp_increase'
        else:
            return 'unknown'
    
    @staticmethod
    def extract_failure_type(filename: str) -> str:
        """Extract failure type from filename."""
        name = Path(filename).stem
        
        if 'bearing_fault' in name:
            return 'bearing_fault'
        elif 'shaft_misalignment' in name:
            return 'shaft_misalignment'
        elif 'rotor_imbalance' in name:
            return 'rotor_imbalance'
        elif 'cavitation' in name:
            return 'cavitation'
        elif 'pipe_blockage' in name:
            return 'pipe_blockage'
        elif 'motor_overload' in name:
            return 'motor_overload'
        elif 'voltage_drop' in name:
            return 'voltage_drop'
        elif 'seal_failure' in name:
            return 'seal_failure'
        elif 'impeller_wear' in name:
            return 'impeller_wear'
        else:
            return 'unknown'
    
    @staticmethod
    def add_hierarchical_labels(df: pd.DataFrame, source_filename: str = None) -> pd.DataFrame:
        """
        Add warning_type and failure_type columns based on health_status.
        
        Args:
            df: DataFrame with health_status column
            source_filename: Original source filename for label extraction
            
        Returns:
            DataFrame with added hierarchical label columns
        """
        df = df.copy()
        
        # Extract labels based on health_status
        if source_filename:
            if 'warning' in source_filename.lower():
                warning_type = LabelExtractor.extract_warning_type(source_filename)
                df[f'warning_type'] = warning_type
            elif 'failure' in source_filename.lower():
                failure_type = LabelExtractor.extract_failure_type(source_filename)
                df[f'failure_type'] = failure_type


        
        return df

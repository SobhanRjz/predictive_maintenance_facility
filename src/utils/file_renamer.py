"""Utility for renaming files and translating column names."""
import shutil
from pathlib import Path
from typing import Dict, List
import pandas as pd
from os import mkdir

def convert_excel_to_csv(source_dir: str, target_dir: str) -> List[str]:
    """
    Convert all Excel files (.csv, .xls) in source_dir to CSV files in target_dir.
    
    Args:
        source_dir: Directory containing Excel files
        target_dir: Directory to save converted CSV files
        
    Returns:
        List of converted file paths
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    converted_files = []
    
    # Find all Excel files
    excel_extensions = ['*.csv', '*.xls']
    excel_files = []
    for ext in excel_extensions:
        excel_files.extend(source_path.glob(ext))
    
    for excel_file in excel_files:
        try:
            # Load Excel file
            df = pd.read_excel(excel_file)
            
            # Determine output filename
            csv_filename = excel_file.stem + '.csv'
            
            csv_path = target_path / csv_filename
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            converted_files.append(str(csv_path))
            
            print(f"Converted {excel_file.name} -> {csv_filename}")
            
        except Exception as e:
            print(f"Error converting {excel_file.name}: {e}")
    
    return converted_files


class FileRenamer:
    """Renames Persian files to English equivalents."""
    
    FILENAME_MAP: Dict[str, str] = {
        # Normal files
        'حالت_نرمال_۳ماهه_گام۳۰ثانیه-1.csv': 'normal_3months_30sec_interval-1.csv',
        'حالت_نرمال_3ماهه_گام30ثانیه-2.csv': 'normal_3months_30sec_interval-2.csv',
        
        # Failure files
        'خرابی1_یاتاقان.csv': 'failure_1_bearing_fault.csv',
        'خرابی2_ناهم محوری محور.csv': 'failure_2_shaft_misalignment.csv',
        'خرابی3_بالانس نبودن روتور.csv': 'failure_3_rotor_imbalance.csv',
        'خرابی4_کاویتاسیون.csv': 'failure_4_cavitation.csv',
        'خرابی5_گرفتگی مسیر.csv': 'failure_5_pipe_blockage.csv',
        'خرابی6_اضافه بار موتور.csv': 'failure_6_motor_overload.csv',
        'خرابی7_افت ولتاژ.csv': 'failure_7_voltage_drop.csv',
        'خرابی8_خرابی سیل.csv': 'failure_8_seal_failure.csv',
        'خرابی9_سایش پروانه.csv': 'failure_9_impeller_wear.csv',
        
        # Warning files
        'هشدار1_افزایش_لرزش_شعاعی.csv': 'warning_1_radial_vibration_increase.csv',
        'هشدار2_افزایش_لرزش_محوری.csv': 'warning_2_axial_vibration_increase.csv',
        'هشدار3_افزایش_دمای_یاتاقان.csv': 'warning_3_bearing_temp_increase.csv',
        'هشدار4_افزایش_دمای_روغن.csv': 'warning_4_oil_temp_increase.csv',
        'هشدار5_افزایش_دمای_پوسته.csv': 'warning_5_casing_temp_increase.csv',
        'هشدار6_افت_فشار_مکش.csv': 'warning_6_suction_pressure_drop.csv',
        'هشدار7_افت_فشار_رانش.csv': 'warning_7_discharge_pressure_drop.csv',
        'هشدار8_کاهش_دبی_جریان.csv': 'warning_8_flow_rate_decrease.csv',
        'هشدار9_افزایش_توان.csv': 'warning_9_power_increase.csv',
        'هشدار10_افزایش_جریان.csv': 'warning_10_current_increase.csv',
        'هشدار11_افت_ولتاژ.csv': 'warning_11_voltage_drop.csv',
        'هشدار12_افزایش_نویز_صوتی.csv': 'warning_12_acoustic_noise_increase.csv',
        'هشدار13_افزایش_دمای_سیال_ خروجی.csv': 'warning_13_outlet_fluid_temp_increase.csv',
        'هشدار14_نوسان_دبی_فشار_توان.csv': 'warning_14_flow_pressure_power_fluctuation.csv',
        'هشدار15_افزایش_لرزش_و_دمای_یاتاقان.csv': 'warning_15_bearing_vibration_temp_increase.csv'
    }
    
    COLUMN_MAP: Dict[str, str] = {
        'زمان': 'timestamp',
        'شتاب‌سنج (g)': 'accelerometer_g',
        'سرعت لرزش (mm/s)': 'vibration_velocity_mm_s',
        'جابجایی محور (µm)': 'shaft_displacement_um',
        'دمای یاتاقان (°C)': 'bearing_temp_c',
        'دمای روغن (°C)': 'oil_temp_c',
        'دمای پوسته (°C)': 'casing_temp_c',
        'دمای سیال ورودی (°C)': 'inlet_fluid_temp_c',
        'دمای سیال خروجی (°C)': 'outlet_fluid_temp_c',
        'فشار ورودی (bar)': 'inlet_pressure_bar',
        'فشار خروجی (bar)': 'outlet_pressure_bar',
        'دبی جریان (m³/h)': 'flow_rate_m3_h',
        'جریان موتور (A)': 'motor_current_a',
        'ولتاژ تغذیه (V)': 'supply_voltage_v',
        'توان مصرفی (kW)': 'power_consumption_kw',
        'شدت صوت (dB)': 'sound_intensity_db',
        'وضعیت سلامت': 'health_status'
    }
    
    def rename_files(self, source_dir: str, target_dir: str) -> None:
        """Copy and rename files from source to target directory."""
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        for persian_name, english_name in self.FILENAME_MAP.items():
            source_file = source_path / persian_name
            df = pd.read_csv(source_file)
            df = self.translate_columns(df)
            df.to_csv(target_path / english_name, index=False)

    
    def translate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translate column names from Persian to English."""
        return df.rename(columns=self.COLUMN_MAP)

def main():
    """Main function to test file renaming."""
    renamer = FileRenamer()
    print("Testing file renaming...")
    print("="*50)
    
    # Test renaming
    target_path = Path('datasets/datasets_renamed')
    target_path.mkdir(parents=True, exist_ok=True)
    renamer.rename_files('datasets', target_path)
    
    print("="*50)
    print("File renaming complete!")

if __name__ == '__main__':
    main()
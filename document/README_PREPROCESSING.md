# Motor Sensor Data Preprocessing - Professional ML Pipeline

## Architecture Overview

```
src/
├── core/              # Interfaces (DI pattern)
├── loaders/           # Data loading with translation
├── filters/           # Condition filtering
├── splitters/         # Time-series splitting strategies
├── exporters/         # Data export
├── validators/        # Data quality checks
├── orchestrator/      # Main pipeline orchestrator
├── factory/           # Dependency injection factory
├── config/            # Configuration management
└── utils/             # File renaming & column translation
```

## Key Features

### 1. **File & Column Translation**
- Persian → English filenames
- Persian → English column names
- Automatic translation during loading

### 2. **Time-Series Aware Splitting**

#### Simple Time-Series Split:
- Chronological train/test split
- No shuffling (preserves temporal order)
- Test set is contiguous suffix

#### Stratified Time-Series Split:
- **RECOMMENDED for your imbalanced data**
- Maintains class distribution
- Preserves temporal order
- Window-based approach prevents data leakage

**Configuration:**
```python
window_size = '10T'  # 10 minutes
# For 1-minute interval data:
# - '5T' = more granular
# - '10T' = balanced (recommended)
# - '30T' = coarser stratification
```

### 3. **Data Quality Validation**
- Missing values detection
- Duplicate timestamps check
- Temporal gaps identification
- Outlier detection (IQR method)
- Numeric range validation

### 4. **Class Imbalance Handling**
Your data characteristics:
- Normal: ~90%
- Warning: ~8%
- Failure: ~2%

Stratified splitting ensures adequate representation of minority classes.

## Usage

### Basic Usage:
```bash
python main.py
```

### Configuration:
Edit `src/config/config.py`:
```python
datasets_dir = 'datasets/datasets_renamed'
output_dir = 'datasets/processed_data'
test_size = 0.2  # 20% for test
use_timeseries_split = True
use_stratification = True  # Recommended for imbalanced data
window_size = '10T'  # 10-minute windows
```

## Output

### Files Generated:
```
datasets/processed_data/
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
└── validation_report.json    # Data quality report
```

### Validation Report Includes:
- Total rows processed
- Missing values count
- Duplicate timestamps
- Temporal gaps
- Outliers per column
- Class distribution

## Column Names (English)

| Persian                    | English                    |
|----------------------------|----------------------------|
| زمان                       | timestamp                  |
| شتاب‌سنج (g)               | accelerometer_g            |
| سرعت لرزش (mm/s)          | vibration_velocity_mm_s    |
| جابجایی محور (µm)          | shaft_displacement_um      |
| دمای یاتاقان (°C)          | bearing_temp_c             |
| دمای روغن (°C)             | oil_temp_c                 |
| دمای پوسته (°C)            | casing_temp_c              |
| دمای سیال ورودی (°C)       | inlet_fluid_temp_c         |
| دمای سیال خروجی (°C)       | outlet_fluid_temp_c        |
| فشار ورودی (bar)           | inlet_pressure_bar         |
| فشار خروجی (bar)           | outlet_pressure_bar        |
| دبی جریان (m³/h)           | flow_rate_m3_h             |
| جریان موتور (A)            | motor_current_a            |
| ولتاژ تغذیه (V)            | supply_voltage_v           |
| توان مصرفی (kW)            | power_consumption_kw       |
| شدت صوت (dB)               | sound_intensity_db         |
| وضعیت سلامت                | health_status              |

## Best Practices for Time-Series ML

### ✅ DO:
1. Use time-series aware splitting
2. Keep temporal order intact
3. Validate data quality before training
4. Handle class imbalance
5. Use stratification for minority classes

### ❌ DON'T:
1. Shuffle time-series data
2. Mix train/test windows
3. Ignore data quality issues
4. Ignore class imbalance
5. Use future data in training

## Why Stratified Time-Series Split?

For motor predictive maintenance:
1. **Temporal Pattern**: Normal → Warning → Failure
2. **Class Imbalance**: Failures are rare (2%)
3. **Real-World**: Test on recent data (simulates production)
4. **No Leakage**: Entire windows to train/test
5. **Balanced Evaluation**: Ensures all classes in test set

## Next Steps

After preprocessing:
1. Feature engineering (rolling statistics, FFT, etc.)
2. Model selection (LSTM, Random Forest, XGBoost)
3. Handle class imbalance in training (SMOTE, class weights)
4. Hyperparameter tuning
5. Cross-validation (time-series CV)


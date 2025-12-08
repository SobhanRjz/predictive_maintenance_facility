# Predictive Maintenance Data Preprocessing

## Overview

The preprocessing pipeline handles raw sensor data transformation for hierarchical classification, implementing automatic feature extraction and hierarchical labeling.

## Unified Config-Driven Pipeline

### Single Script for All Training
```bash
# Layer 1: Full preprocessing + training
python main_ml_pipeline.py

# Layer 2: Uses preprocessed data + filtering
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml
```

## Data Flow Architecture

### Layer 1: Complete Preprocessing Pipeline
```
Raw CSV Files (30sec intervals)
    ↓
MultiFileLoader
├── Load CSV/Excel files
├── Translate Persian → English (if needed)
├── Resample: 30sec → 1min
└── Add run_id column
    ↓
PreprocessingOrchestrator
├── Filter normal/abnormal data
├── TimeDomainFeatureExtractor (10min windows)
│   ├── Statistical: mean, std, rms, peak-to-peak
│   ├── Shape: skewness, kurtosis, crest factor
│   └── Distribution: percentiles, variance
├── Hierarchical Label Extractor
│   ├── warning_type from filename
│   └── failure_type from filename
├── RunIdSplitter (prevents data leakage)
├── DataValidator (quality checks)
└── CSVExporter
    ↓
Processed Data (train.csv, test.csv)
    ↓
MLOrchestrator + XGBoost Training
```

### Layer 2: Filtered Training Pipeline
```
Preprocessed Data (train.csv, test.csv)
    ↓
DataFilter (health_status == "Warning")
    ↓
MLOrchestrator + XGBoost Training
```

## Configuration Structure

### Layer 1 Config (Full Preprocessing)
```yaml
dataset:
  datasets_dir: 'datasets\datasets_renamed'
  output_dir: 'datasets\processed_data'
  normal_files:
    - 'normal_3months_30sec_interval-1.csv'
  failure_files:
    - 'failure_1_bearing_fault.csv'
    # ... more files
  warning_files:
    - 'warning_1_radial_vibration_increase.csv'
    # ... more files

  # Preprocessing settings
  resample_freq: "1min"        # 30sec → 1min
  add_hierarchical_labels: true

features:
  window_size: "10min"         # Rolling window size
  extract_features: true       # Enable feature extraction
  timestamp_col: "timestamp"
  exclude_cols: ["run_id"]     # Columns to exclude from features
```

### Layer 2 Config (Uses Processed Data)
```yaml
dataset:
  output_dir: 'datasets\processed_data'  # Uses processed data
  # No file lists needed

features:
  extract_features: false      # Features already extracted

data_filter:                   # Filter to specific samples
  column: "health_status"
  operator: "=="
  value: "Warning"
```

## Feature Extraction Details

### Time-Domain Features (10-minute windows)
- **Basic Statistics**: mean, std, min, max, rms, peak-to-peak
- **Shape Features**: skewness, kurtosis, crest factor
- **Distribution**: percentiles (25%, 50%, 75%), variance
- **Advanced**: zero crossings, peak count, energy

### Rolling Window Implementation
```python
# 10-minute windows on 1-minute data = 10 samples per window
window_size = "10min"
features_per_sensor = 15+  # Statistical features
total_features = features_per_sensor × num_sensors
```

## Hierarchical Labeling System

### Automatic Label Extraction
Labels extracted from filenames during preprocessing:

```
warning_1_radial_vibration_increase.csv → warning_type: "radial_vibration_increase"
failure_2_shaft_misalignment.csv → failure_type: "shaft_misalignment"
```

### Warning Types (15 classes)
- `radial_vibration_increase`
- `axial_vibration_increase`
- `bearing_temp_increase`
- `oil_temp_increase`
- `casing_temp_increase`
- `suction_pressure_drop`
- `discharge_pressure_drop`
- `flow_rate_decrease`
- `power_increase`
- `current_increase`
- `voltage_drop`
- `acoustic_noise_increase`
- `outlet_fluid_temp_increase`
- `flow_pressure_power_fluctuation`
- `bearing_vibration_temp_increase`

### Failure Types (9 classes)
- `bearing_fault`
- `shaft_misalignment`
- `rotor_imbalance`
- `cavitation`
- `pipe_blockage`
- `motor_overload`
- `seal_failure`
- `impeller_wear`

## Data Splitting Strategy

### Run-Based Splitting (Recommended)
- **Purpose**: Prevents data leakage between train/test
- **Method**: Split by `run_id` (each CSV file = 1 run)
- **Benefits**:
  - Maintains temporal sequences within runs
  - No future data in training set
  - Realistic evaluation scenario

### Configuration
```yaml
dataset:
  use_timeseries_split: false    # Use run-based splitting
  use_stratification: false      # No stratification needed
  test_size: 0.2                # 20% test set
```

## Data Quality Validation

### Automatic Checks
- Missing values detection
- Duplicate timestamps
- Outlier detection
- Data type consistency
- Column completeness

### Validation Reports
```json
{
  "missing_values": {"column": "sensor_1", "count": 5},
  "duplicate_timestamps": 0,
  "outliers_detected": 12,
  "data_types_valid": true
}
```

## Memory & Performance Optimization

### Streaming Processing
- Large CSV files processed in chunks
- Feature extraction batched by time windows
- Memory-efficient pandas operations

### Parallel Processing
- Multiple files loaded concurrently
- Feature extraction parallelized
- GPU acceleration for XGBoost (optional)

## Usage Examples

### Process & Train Layer 1
```bash
python main_ml_pipeline.py
```

**Creates**:
- `datasets/processed_data/train.csv` (with features + labels)
- `datasets/processed_data/test.csv`
- `models/layer1_anomaly_detection.pkl`

### Train Layer 2 Classifiers
```bash
# Warning classifier
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml

# Failure classifier
python main_ml_pipeline.py configs/model_configs/layer2_failure_classifier.yaml
```

## Configuration Customization

### Change Window Size
```yaml
features:
  window_size: "5min"    # More granular (5min windows)
  # or "15min"          # Coarser (15min windows)
```

### Add New Sensors
```yaml
# Automatically detected from CSV columns
# No config changes needed
```

### Modify Feature Set
```python
# Extend TimeDomainFeatureExtractor class
def extract_additional_features(self, df):
    # Add custom features
    df['custom_feature'] = df['sensor_1'] * df['sensor_2']
    return df
```

### Custom Resampling
```yaml
dataset:
  resample_freq: "30sec"    # Keep original sampling
  # or "2min"              # Coarser sampling
```

## Output Data Format

### Processed CSV Structure
```
timestamp,run_id,health_status,warning_type,failure_type,
sensor1_mean,sensor1_std,sensor1_rms,...,
sensor2_mean,sensor2_std,sensor2_rms,...,
...
```

- **timestamp**: Datetime index
- **run_id**: File identifier (prevents leakage)
- **health_status**: Normal/Warning/Failure
- **warning_type**: Specific warning class (if Warning)
- **failure_type**: Specific failure class (if Failure)
- **Features**: Time-domain features per sensor

## Troubleshooting

### Memory Issues
- Reduce window size: `window_size: "5min"`
- Increase batch size in feature extraction
- Use data sampling for development

### Slow Processing
- Enable parallel processing in config
- Use coarser resampling: `resample_freq: "2min"`
- Reduce feature set complexity

### Label Extraction Issues
- Check filename format matches expected patterns
- Verify Persian→English translation working
- Review label extraction logic in `label_extractor.py`

### Data Quality Issues
- Check validation reports in logs
- Handle missing values appropriately
- Review outlier detection thresholds

## Extensibility

### Add New Feature Types
1. Implement `IFeatureExtractor` interface
2. Add to `OrchestratorFactory.create()`
3. Update config: `feature_types: ["time_domain", "frequency"]`

### Custom Labeling
1. Extend `LabelExtractor` class
2. Add filename pattern matching
3. Update config target classes

### New Data Sources
1. Implement `IDataLoader` interface
2. Add to `MultiFileLoader`
3. Update file format detection
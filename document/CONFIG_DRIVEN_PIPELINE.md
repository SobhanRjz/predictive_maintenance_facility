# Unified Config-Driven ML Pipeline

## Overview

**Single script (`main_ml_pipeline.py`) + Layer configs = Complete hierarchical ML pipeline**

No more separate scripts - everything runs through the unified pipeline with different YAML configurations.

## Architecture

### Before (Complex)
```
train_model_from_config.py + model configs
main_ml_pipeline.py + pipeline_config.yaml
├── train_model_from_config.py (Layer 2)
├── main_ml_pipeline.py (Layer 1)
└── configs/pipeline_config.yaml (separate config)
```

### After (Unified)
```
main_ml_pipeline.py + layer configs
├── main_ml_pipeline.py (all layers)
└── configs/model_configs/*.yaml (all configs)
```

## Usage

### Train Layer 1 (Anomaly Detection)
```bash
python main_ml_pipeline.py
# Uses: configs/model_configs/layer1_anomaly_detection.yaml
```

### Train Layer 2 (Warning Types)
```bash
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml
```

### Train Layer 2 (Failure Types)
```bash
python main_ml_pipeline.py configs/model_configs/layer2_failure_classifier.yaml
```

## Configuration Structure

### Layer 1 Config (`layer1_anomaly_detection.yaml`)
```yaml
model:
  name: "layer1_anomaly_detection"
  type: "xgboost"

  # Complete dataset specification
  dataset:
    datasets_dir: 'datasets\datasets_renamed'
    output_dir: 'datasets\processed_data'
    normal_files: ['normal_3months_30sec_interval-1.csv']
    failure_files: ['failure_1_bearing_fault.csv', ...]
    warning_files: ['warning_1_radial_vibration_increase.csv', ...]
    resample_freq: "1min"
    add_hierarchical_labels: true

  # Feature extraction settings
  features:
    window_size: "10min"
    extract_features: true
    timestamp_col: "timestamp"
    exclude_cols: ["run_id"]

  # Model configuration
  target:
    column: "health_status"
    classes: ["Normal", "Warning", "Failure"]

  hyperparameters:
    n_estimators: 1000
    max_depth: 8
    scale_pos_weight: [1, 28, 62]

  # No filtering (uses all data)
  data_filter: null

  # Output paths
  output:
    model_path: "models/layer1_anomaly_detection.pkl"
    metrics_path: "outputs/metrics/layer1_anomaly_detection.json"
```

### Layer 2 Config (`layer2_warning_classifier.yaml`)
```yaml
model:
  name: "layer2_warning_classifier"
  type: "xgboost"

  # Uses preprocessed data (no file lists)
  dataset:
    datasets_dir: 'datasets\datasets_renamed'
    output_dir: 'datasets\processed_data'
    resample_freq: "1min"
    add_hierarchical_labels: false

  # Feature settings (uses preprocessed features)
  features:
    window_size: "10min"
    extract_features: false  # Already done in Layer 1
    timestamp_col: "timestamp"
    exclude_cols: ["run_id", "health_status"]

  # Different target for Layer 2
  target:
    column: "warning_type"
    classes: ["radial_vibration_increase", "bearing_temp_increase", ...]

  hyperparameters:
    n_estimators: 800
    max_depth: 7

  # Filter to Warning samples only
  data_filter:
    column: "health_status"
    operator: "=="
    value: "Warning"

  # Output paths
  output:
    model_path: "models/layer2_warning_classifier.pkl"
    metrics_path: "outputs/metrics/layer2_warning_classifier.json"
```

## Intelligent Pipeline Execution

### Layer 1 Detection
```python
# Has data files → Full preprocessing pipeline
if len(dataset_config.normal_files) > 0 or len(dataset_config.failure_files) > 0:
    # Load raw data, preprocess, extract features, train
    preprocess_and_train()
```

### Layer 2 Detection
```python
# No data files → Load preprocessed data, filter, train
else:
    # Load train.csv/test.csv, apply filter, train
    load_filter_train()
```

## Workflows

### Complete Pipeline (All Layers)
```bash
# 1. Train Layer 1 (creates processed data)
python main_ml_pipeline.py

# 2. Train Layer 2a (Warning classifier)
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml

# 2. Train Layer 2b (Failure classifier)
python main_ml_pipeline.py configs/model_configs/layer2_failure_classifier.yaml
```

### Retrain Individual Layers
```bash
# Edit any config YAML file
# Change hyperparameters, features, etc.

# Retrain specific layer
python main_ml_pipeline.py configs/model_configs/layer1_anomaly_detection.yaml
```

### Experiment with Different Settings
```bash
# Create experiment config
cp configs/model_configs/layer1_anomaly_detection.yaml \
   configs/model_configs/layer1_experiment1.yaml

# Edit experiment1.yaml (change n_estimators: 2000, max_depth: 12)

# Run experiment
python main_ml_pipeline.py configs/model_configs/layer1_experiment1.yaml
```

## Key Benefits

### ✅ Single Script Architecture
- One `main_ml_pipeline.py` for everything
- No confusion about which script to use
- Consistent interface across all layers

### ✅ Unified Configuration
- All configs follow same structure
- Layer 1: Includes data files + preprocessing
- Layer 2: Uses preprocessed data + filtering
- Easy to understand and modify

### ✅ Automatic Pipeline Selection
- Script detects Layer 1 vs Layer 2 automatically
- No manual pipeline selection
- Prevents configuration errors

### ✅ Hierarchical Labels Built-in
- Automatic extraction during Layer 1
- Available for Layer 2 without reprocessing
- Consistent labeling across experiments

### ✅ Flexible Data Handling
- Layer 1: Processes raw files from disk
- Layer 2: Uses cached processed data
- Efficient for iterative development

## Migration from Old System

### Old Way (2 Scripts + 2 Config Types)
```bash
# Train Layer 1
python main_ml_pipeline.py  # Uses hardcoded configs

# Train Layer 2
python train_model_from_config.py layer2_warning_classifier.yaml
```

### New Way (1 Script + 1 Config Type)
```bash
# Train Layer 1
python main_ml_pipeline.py  # Uses layer1_anomaly_detection.yaml

# Train Layer 2
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml
```

## Configuration Examples

### Change Hyperparameters
```yaml
hyperparameters:
  n_estimators: 1500      # From 1000
  max_depth: 10           # From 8
  learning_rate: 0.03     # From 0.05
  subsample: 0.8          # Add regularization
```

### Add New Data Files
```yaml
dataset:
  warning_files:
    - 'warning_16_new_sensor.csv'
    - 'warning_17_another_type.csv'
```

### Change Feature Extraction
```yaml
features:
  window_size: "5min"      # From 10min
  extract_features: true
  exclude_cols: ["run_id", "timestamp"]  # Add timestamp exclusion
```

### Filter Different Data Subsets
```yaml
data_filter:
  column: "warning_type"
  operator: "in"
  value: ["vibration", "temperature"]  # Only vibration/temperature warnings
```

## File Organization

```
configs/model_configs/
├── layer1_anomaly_detection.yaml    # Full pipeline config
├── layer2_warning_classifier.yaml   # Warning classifier config
└── layer2_failure_classifier.yaml   # Failure classifier config

main_ml_pipeline.py                  # Unified training script

datasets/
├── datasets_renamed/               # Raw CSV files (Layer 1 input)
└── processed_data/                 # train.csv, test.csv (Layer 2 input)

models/                             # Trained models output
outputs/metrics/                    # Performance metrics output
```

## Troubleshooting

### "Configuration file not found"
```bash
python main_ml_pipeline.py configs/model_configs/layer1_anomaly_detection.yaml
```

### "Preprocessed data not found" (Layer 2)
- Run Layer 1 first: `python main_ml_pipeline.py`
- Creates `datasets/processed_data/train.csv` and `test.csv`

### "No data files specified"
- For Layer 1: Add `normal_files`, `failure_files`, `warning_files`
- For Layer 2: Leave empty (uses processed data)

### YAML syntax errors
- Validate with online YAML validator
- Check indentation (2 spaces per level)
- Quote strings with special characters

## Performance Optimizations

### Memory Efficiency
- Layer 1: Streams raw data processing
- Layer 2: Loads cached processed data
- Feature extraction in batches

### Training Efficiency
- Early stopping prevents overfitting
- Class balancing handles imbalanced data
- Stratified sampling for Layer 2

### Development Efficiency
- Quick Layer 2 iterations (no reprocessing)
- Independent layer retraining
- Config-based experimentation

## Future Extensions

### Add New Layers
1. Create new config with appropriate `data_filter`
2. Define target column and classes
3. Run with unified script

### Add New Models
1. Implement `IModel` interface
2. Add to `ModelFactory.create()`
3. Set `type: "new_model"` in config

### Add New Features
1. Implement `IFeatureExtractor`
2. Add to feature extraction pipeline
3. Update `feature_types` in config
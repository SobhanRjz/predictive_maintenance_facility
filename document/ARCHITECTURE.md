# Hierarchical ML Architecture

## Overview

This architecture implements a hierarchical classification system for predictive maintenance, using a unified config-driven approach that eliminates code changes for model retraining.

## Hierarchical Classification System

```
Layer 1: Anomaly Detection
├── Input: Raw sensor data
├── Output: Normal | Warning | Failure
├── Config: layer1_anomaly_detection.yaml
└── Features: Full preprocessing + feature extraction

Layer 2a: Warning Classification
├── Input: Warning samples (filtered from Layer 1)
├── Output: 15 specific warning types
├── Config: layer2_warning_classifier.yaml
└── Features: Uses Layer 1 preprocessed features

Layer 2b: Failure Classification
├── Input: Failure samples (filtered from Layer 1)
├── Output: 9 specific failure types
├── Config: layer2_failure_classifier.yaml
└── Features: Uses Layer 1 preprocessed features
```

## Unified Config-Driven Architecture

### Single Script, Multiple Configs

**`main_ml_pipeline.py`** - One script handles all training:

```bash
# Layer 1 (full pipeline)
python main_ml_pipeline.py

# Layer 2 (uses preprocessed data)
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml
```

### Configuration Structure

Each YAML config contains complete pipeline definition:

```yaml
model:
  name: "layer1_anomaly_detection"
  type: "xgboost"

  # Dataset & preprocessing
  dataset:
    datasets_dir: 'datasets\datasets_renamed'
    normal_files: ['normal_3months_30sec_interval-1.csv']
    failure_files: ['failure_1_bearing_fault.csv', ...]
    resample_freq: "1min"
    add_hierarchical_labels: true

  # Feature extraction
  features:
    window_size: "10min"
    extract_features: true
    timestamp_col: "timestamp"
    exclude_cols: ["run_id"]

  # Target definition
  target:
    column: "health_status"
    classes: ["Normal", "Warning", "Failure"]

  # Model hyperparameters
  hyperparameters:
    n_estimators: 1000
    max_depth: 8
    learning_rate: 0.05
    scale_pos_weight: [1, 28, 62]

  # Data filtering (Layer 2 only)
  data_filter: null

  # Output paths
  output:
    model_path: "models/layer1_anomaly_detection.pkl"
    metrics_path: "outputs/metrics/layer1_anomaly_detection.json"
```

## Architecture Components

### 1. Core Interfaces (`src/ML/core/interfaces.py`)

- `IDataLoader` - Data loading strategies (CSV, Excel)
- `IDataFilter` - Data filtering strategies
- `IDataSplitter` - Train/test splitting strategies
- `IDataExporter` - Data export strategies
- `IFeatureExtractor` - Feature extraction strategies
- `IModel` - ML/DL model interface

### 2. Feature Extraction (`src/ML/features/`)

- `TimeDomainFeatureExtractor` - Statistical time-domain features
  - Basic: mean, std, min, max, rms, peak-to-peak
  - Advanced: kurtosis, skewness, crest factor, percentiles
- Extensible for: frequency domain, wavelet transforms, etc.

### 3. Models (`src/ML/models/`)

- `XGBoostModel` - Gradient boosting classifier with class balancing
- `BaseDLModel` - Base class for future DL models (LSTM, CNN, Transformer)

### 4. Orchestrators (`src/ML/orchestrator/`)

- `PreprocessingOrchestrator` - Data preprocessing pipeline
- `MLOrchestrator` - ML training & evaluation pipeline

### 5. Config System (`src/ML/config/`)

- `config_loader.py` - YAML config parser into dataclasses
- Unified configuration for dataset, features, model, and output

### 6. Data Processing Pipeline

```
Raw CSV Files
    ↓
MultiFileLoader (CSV/Excel + Resampling)
    ↓
PreprocessingOrchestrator
├── Filters (normal/abnormal separation)
├── Feature Extractor (time-domain features)
├── Hierarchical Labels (warning_type/failure_type)
├── Splitter (run-based to prevent leakage)
└── Exporter (train.csv, test.csv)
    ↓
MLOrchestrator
├── Model Factory (XGBoost)
├── Training (with class balancing)
├── Evaluation (accuracy, precision, recall, F1)
└── Export (model.pkl, metrics.json)
```

## Design Patterns

### Strategy Pattern

All components implement interfaces, allowing runtime algorithm selection:

```python
# Different feature extractors
feature_extractor: IFeatureExtractor = TimeDomainFeatureExtractor()

# Different models
model: IModel = XGBoostModel()  # Future: LSTMModel(), CNNModel()
```

### Factory Pattern

Centralized creation of complex objects:

```python
# Orchestrator factory creates complete pipeline
orchestrator = OrchestratorFactory.create(
    use_timeseries_split=False,
    window_size="10min",
    extract_features=True
)

# Model factory creates configured models
model = ModelFactory.create(config)
```

### Config-Driven Pattern

YAML configurations eliminate code changes:

```yaml
# Change hyperparameters without code modification
hyperparameters:
  n_estimators: 2000  # Was 1000
  max_depth: 12       # Was 8
```

## Data Flow Architecture

### Layer 1 (Full Pipeline)

```
Raw Files → Load → Resample → Features → Labels → Split → Train Model
```

1. **Load**: CSV files with 30sec intervals
2. **Resample**: Aggregate to 1min intervals
3. **Features**: 10min rolling window time-domain features
4. **Labels**: Extract warning_type/failure_type from filenames
5. **Split**: Run-based splitting (prevents data leakage)
6. **Train**: XGBoost with class balancing

### Layer 2 (Filtered Training)

```
Preprocessed Data → Filter → Train Model
```

1. **Load**: train.csv/test.csv from Layer 1
2. **Filter**: Only Warning samples (for warning classifier)
3. **Train**: XGBoost on specific subset

## Hierarchical Inference

Conditional execution based on Layer 1 predictions:

```python
predictor = HierarchicalPredictor(
    layer1_model="models/layer1_anomaly_detection.pkl",
    layer2_warning_model="models/layer2_warning_classifier.pkl",
    layer2_failure_model="models/layer2_failure_classifier.pkl"
)

# Single prediction call handles all layers
results = predictor.predict(X)
# Returns: health_status + warning_type (if Warning) + failure_type (if Failure)
```

## Extensibility

### Adding New Models

1. Implement `IModel` interface
2. Add to `ModelFactory.create()`
3. Update YAML config: `type: "new_model"`

### Adding New Features

1. Implement `IFeatureExtractor` interface
2. Add to `OrchestratorFactory.create()`
3. Update YAML config: `feature_types: ["new_features"]`

### Adding New Layers

1. Create new YAML config with appropriate filters
2. Define target column and classes
3. Run: `python main_ml_pipeline.py configs/model_configs/new_layer.yaml`

## Benefits

### 1. No Code Changes for Retraining
- Edit YAML configs
- Run same script
- Compare results

### 2. Hierarchical Labels
- Automatic extraction from filenames
- Enables multi-layer classification
- Consistent labeling across experiments

### 3. Flexible Data Filtering
- Layer 2 trains on relevant subsets
- Easy experimentation with different subsets
- Prevents information leakage

### 4. Independent Layer Training
- Train layers separately
- Mix and match layer versions
- Easy debugging and optimization

### 5. Version Control Friendly
- Config changes tracked in git
- Reproducible experiments
- Easy rollback

## Configuration Examples

### Experiment with Hyperparameters

```yaml
# configs/model_configs/layer1_anomaly_detection.yaml
hyperparameters:
  n_estimators: 2000      # Increase iterations
  max_depth: 10           # Deeper trees
  learning_rate: 0.03     # Slower learning
  subsample: 0.8          # Add randomness
```

### Train on Specific Subsets

```yaml
# Layer 2 config
data_filter:
  column: "health_status"
  operator: "=="
  value: "Warning"
```

### Add New Data Files

```yaml
# Layer 1 config
dataset:
  warning_files:
    - 'warning_16_new_sensor.csv'
    - 'warning_17_another_type.csv'
```

## Performance Considerations

### Memory Management
- Streaming data loading for large datasets
- Feature extraction in batches
- Model training with early stopping

### Training Optimization
- Class balancing with `scale_pos_weight`
- Stratified sampling for Layer 2
- Cross-validation support (future)

### Inference Optimization
- Conditional Layer 2 execution
- Feature preprocessing caching
- Model serialization optimization
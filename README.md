# Predictive Maintenance Facility - Hierarchical ML Pipeline

## Overview

A complete machine learning pipeline for predictive maintenance in industrial facilities, featuring a hierarchical classification architecture that first detects anomalies (Normal/Warning/Failure) and then classifies specific warning or failure types.

## Architecture

### Hierarchical Classification System

```
Layer 1: Anomaly Detection
├── Input: Raw sensor data (vibration, temperature, pressure, etc.)
├── Output: Normal | Warning | Failure
└── Model: XGBoost classifier

Layer 2a: Warning Type Classification (if Warning detected)
├── Input: Warning samples from Layer 1
├── Output: 15 specific warning types (vibration, temperature, pressure, etc.)
└── Model: XGBoost classifier

Layer 2b: Failure Type Classification (if Failure detected)
├── Input: Failure samples from Layer 1
├── Output: 9 specific failure types (bearing, shaft, cavitation, etc.)
└── Model: XGBoost classifier
```

## Quick Start

### 1. Train Layer 1 (Anomaly Detection)
```bash
python main_ml_pipeline.py
```

### 2. Train Layer 2 (Warning Types)
```bash
python main_ml_pipeline.py configs/model_configs/layer2_warning_classifier.yaml
```

### 3. Train Layer 2 (Failure Types)
```bash
python main_ml_pipeline.py configs/model_configs/layer2_failure_classifier.yaml
```

## Project Structure

```
├── configs/
│   └── model_configs/
│       ├── layer1_anomaly_detection.yaml      # Layer 1 config
│       ├── layer2_warning_classifier.yaml     # Layer 2a config
│       └── layer2_failure_classifier.yaml     # Layer 2b config
├── datasets/
│   ├── datasets_renamed/                       # Raw CSV files
│   └── processed_data/                         # Preprocessed data
├── models/                                     # Trained models
├── outputs/
│   ├── metrics/                                # Performance metrics
│   └── predictions/                            # Model predictions
├── src/ML/
│   ├── config/                                 # Configuration loaders
│   ├── core/                                   # Interfaces
│   ├── features/                               # Feature extraction
│   ├── filters/                                # Data filtering
│   ├── loaders/                                # Data loading
│   ├── models/                                 # ML models
│   ├── orchestrator/                           # Pipeline orchestrators
│   ├── splitters/                              # Data splitting
│   ├── trainers/                               # Training utilities
│   └── utils/                                  # Utilities
└── document/                                   # Documentation
```

## Configuration-Driven Architecture

### Key Features

- **No Code Changes**: Modify YAML configs to retrain models with different hyperparameters
- **Hierarchical Labels**: Automatically extracted from filenames during preprocessing
- **Flexible Data Filtering**: Layer 2 models train only on relevant subsets
- **Conditional Inference**: Layer 2 runs only when needed
- **Independent Layer Training**: Train any layer without affecting others

### Configuration Files

Each model config contains:
- **Dataset**: File paths, preprocessing settings, splitting strategy
- **Features**: Window size, extraction methods, column exclusions
- **Target**: Column name, classes, classification type
- **Hyperparameters**: Model-specific parameters (n_estimators, max_depth, etc.)
- **Data Filter**: Optional filtering for hierarchical training
- **Output**: Model and metrics file paths

## Usage Examples

### Train Layer 1 with Different Hyperparameters
```bash
# Edit configs/model_configs/layer1_anomaly_detection.yaml
# Change n_estimators: 2000, max_depth: 10, etc.

python main_ml_pipeline.py configs/model_configs/layer1_anomaly_detection.yaml
```

### Train Only on Specific Failure Types
```yaml
# configs/model_configs/layer2_failure_classifier.yaml
data_filter:
  column: "health_status"
  operator: "=="
  value: "Failure"
```

### Add New Warning Types
```yaml
# configs/model_configs/layer2_warning_classifier.yaml
dataset:
  warning_files:
    - 'warning_16_new_type.csv'  # Add new file
target:
  classes:
    - "new_warning_type"         # Add new class
```

## Hierarchical Inference

After training all layers:

```python
from src.ML.inference.hierarchical_predictor import HierarchicalPredictor

predictor = HierarchicalPredictor(
    layer1_model_path="models/layer1_anomaly_detection.pkl",
    layer2_warning_model_path="models/layer2_warning_classifier.pkl",
    layer2_failure_model_path="models/layer2_failure_classifier.pkl"
)

results = predictor.predict(X)
# Returns: health_status, warning_type, failure_type
```

## Data Pipeline

### Input Data Format
- **Raw Files**: CSV files with sensor readings (vibration, temperature, pressure, etc.)
- **Sampling**: 30-second intervals (resampled to 1-minute)
- **Labels**: Extracted from filenames (warning_1_vibration.csv → warning_type: vibration)

### Preprocessing Steps
1. **Load & Resample**: 30sec → 1min intervals
2. **Feature Extraction**: Time-domain features (mean, std, rms, kurtosis, etc.)
3. **Hierarchical Labels**: Extract warning_type/failure_type from filenames
4. **Train/Test Split**: Run-based splitting to prevent data leakage
5. **Export**: Processed train.csv and test.csv

## Model Details

### XGBoost Configuration
- **Layer 1**: Imbalanced classes (scale_pos_weight: [1, 28, 62])
- **Layer 2**: Balanced classes (stratified sampling)
- **Early Stopping**: Prevents overfitting
- **Evaluation**: Multi-class log loss

### Feature Engineering
- **Time Windows**: 10-minute rolling windows
- **Time-Domain Features**:
  - Statistical: mean, std, min, max, rms, peak-to-peak
  - Shape: skewness, kurtosis, crest factor
  - Distribution: percentiles, variance

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- pyyaml >= 6.0.0

## Documentation

- [`document/ARCHITECTURE.md`](document/ARCHITECTURE.md) - Detailed architecture overview
- [`document/CONFIG_DRIVEN_PIPELINE.md`](document/CONFIG_DRIVEN_PIPELINE.md) - Config system details
- [`document/README_PREPROCESSING.md`](document/README_PREPROCESSING.md) - Data preprocessing guide
- [`document/SPLITTING_STRATEGY_ANALYSIS.md`](document/SPLITTING_STRATEGY_ANALYSIS.md) - Data splitting analysis

## Troubleshooting

### "Configuration file not found"
```bash
python main_ml_pipeline.py configs/model_configs/layer1_anomaly_detection.yaml
```

### "Preprocessed data not found" (Layer 2)
- Run Layer 1 first: `python main_ml_pipeline.py`
- This creates `datasets/processed_data/train.csv` and `test.csv`

### YAML syntax errors
- Use online YAML validator
- Check indentation (spaces, not tabs)
- Ensure proper quoting for strings

### Import errors
```bash
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes to YAML configs (preferred) or code
4. Test with existing pipeline
5. Submit pull request

## License

[Add license information here]
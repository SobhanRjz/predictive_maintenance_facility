# Extended ML Architecture

## Overview

This architecture extends the preprocessing pipeline with ML/DL capabilities while maintaining modularity and extensibility.

## Architecture Layers

### 1. Core Interfaces (`src/core/interfaces.py`)
- `IDataLoader` - Data loading strategies
- `IDataFilter` - Data filtering strategies
- `IDataSplitter` - Train/test splitting strategies
- `IDataExporter` - Data export strategies
- `IFeatureExtractor` - Feature extraction strategies (NEW)
- `IModel` - ML/DL model interface (NEW)

### 2. Feature Extraction (`src/features/`)
- `TimeDomainFeatureExtractor` - Statistical time-domain features
  - Mean, Std, Min, Max, RMS, Peak-to-Peak, Kurtosis, Skewness
- Future: Frequency domain, wavelet, etc.

### 3. Models (`src/models/`)
- `XGBoostModel` - Gradient boosting classifier
- `BaseDLModel` - Base class for future DL models (LSTM, CNN, Transformer)

### 4. Orchestrators (`src/orchestrator/`)
- `PreprocessingOrchestrator` - Data preprocessing pipeline
- `MLOrchestrator` - ML training & evaluation pipeline (NEW)

### 5. Factories (`src/factory/`)
- `OrchestratorFactory` - Creates preprocessing orchestrator
- `MLOrchestratorFactory` - Creates ML orchestrator (NEW)

### 6. Configuration (`src/config/`)
- `DatasetConfig` - Preprocessing configuration
- `MLConfig` - ML training configuration (NEW)

## Design Patterns

### Strategy Pattern
All components implement interfaces, allowing runtime algorithm selection:
```python
# Different feature extractors
feature_extractor: IFeatureExtractor = TimeDomainFeatureExtractor()

# Different models
model: IModel = XGBoostModel()  # or future: LSTMModel(), CNNModel()
```

### Factory Pattern
Centralized object creation with dependency injection:
```python
ml_orchestrator = MLOrchestratorFactory.create(
    feature_window_size='10T',
    n_estimators=100
)
```

### Orchestrator Pattern
High-level workflow coordination:
```python
# Preprocessing
train_df, test_df, _ = preprocessing_orchestrator.process(...)

# ML Training
metrics = ml_orchestrator.train_and_evaluate(train_df, test_df)
```

## Extensibility

### Adding New Feature Extractors
```python
class FrequencyDomainFeatureExtractor(IFeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        # FFT-based features
        pass
```

### Adding New Models
```python
class LSTMModel(IModel):
    def train(self, X_train, y_train):
        # LSTM training
        pass
```

### Adding New Orchestrators
```python
class DLOrchestrator:
    def __init__(self, feature_extractor, model):
        # Sequence-based DL pipeline
        pass
```

## Pipeline Flow

```
Raw Data
    ↓
PreprocessingOrchestrator
    ├── Load (MultiFileLoader)
    ├── Filter (IncludeOnlyNormalFilter, ExcludeNormalFilter)
    ├── Resample (DataResampler)
    ├── Validate (DataValidator)
    ├── Split (StratifiedTimeSeriesSplitter)
    └── Export (CSVExporter)
    ↓
Train/Test DataFrames
    ↓
MLOrchestrator
    ├── Extract Features (TimeDomainFeatureExtractor)
    ├── Train Model (XGBoostModel)
    ├── Evaluate (Metrics)
    └── Save Model
    ↓
Trained Model + Metrics
```

## Usage

### Complete Pipeline
```python
from src.factory.orchestrator_factory import OrchestratorFactory
from src.factory.ml_orchestrator_factory import MLOrchestratorFactory

# Preprocessing
prep_orch = OrchestratorFactory.create()
train_df, test_df, _ = prep_orch.process(...)

# ML Training
ml_orch = MLOrchestratorFactory.create()
metrics = ml_orch.train_and_evaluate(train_df, test_df)
```

## Future Extensions

1. **Frequency Domain Features** - FFT, PSD, spectral features
2. **Deep Learning Models** - LSTM, CNN, Transformer for sequence modeling
3. **Ensemble Methods** - Combine multiple models
4. **Hyperparameter Tuning** - Grid search, Bayesian optimization
5. **Online Learning** - Incremental model updates
6. **Model Serving** - REST API for predictions


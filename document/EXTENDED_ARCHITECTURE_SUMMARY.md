# Extended Architecture Summary

## New Components Added

### 1. Feature Extraction Layer
**Location**: `src/features/`
- `TimeDomainFeatureExtractor` - Extracts 8 statistical features per sensor
  - Mean, Std, Min, Max, RMS, Peak-to-Peak, Kurtosis, Skewness
- Implements `IFeatureExtractor` interface for extensibility

### 2. Model Layer
**Location**: `src/models/`
- `XGBoostModel` - Gradient boosting classifier with scikit-learn API
- `BaseDLModel` - Abstract base for future DL implementations (LSTM, CNN, etc.)
- Both implement `IModel` interface

### 3. ML Orchestrator
**Location**: `src/orchestrator/ml_orchestrator.py`
- Coordinates: Feature Extraction → Training → Evaluation → Model Saving
- Works with any `IFeatureExtractor` and `IModel` implementation

### 4. ML Factory
**Location**: `src/factory/ml_orchestrator_factory.py`
- Creates ML orchestrator with configured dependencies
- Currently wires: `TimeDomainFeatureExtractor` + `XGBoostModel`

### 5. ML Configuration
**Location**: `src/config/config.py`
- `MLConfig` dataclass for ML hyperparameters
- Separate from preprocessing config for modularity

### 6. Main Pipeline Script
**Location**: `main_ml_pipeline.py`
- End-to-end pipeline: Preprocessing → Feature Extraction → Training

## Updated Components

### Core Interfaces (`src/core/interfaces.py`)
Added two new interfaces:
- `IFeatureExtractor` - Feature extraction strategy
- `IModel` - ML/DL model interface (train, predict, evaluate, save, load)

### Requirements (`requirements.txt`)
Added:
- `xgboost>=2.0.0`
- `joblib>=1.3.0`

## Architecture Benefits

1. **Minimal & Focused**: Only XGBoost + time-domain features as requested
2. **Extensible**: Easy to add new feature extractors or models via interfaces
3. **DL-Ready**: `BaseDLModel` provides foundation for future DL models
4. **Consistent Design**: Follows existing patterns (Strategy, Factory, Orchestrator)
5. **Production-Ready**: Clean separation of concerns, type hints, docstrings

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main_ml_pipeline.py
```

## File Structure

```
src/
├── core/
│   └── interfaces.py          [UPDATED: +IFeatureExtractor, +IModel]
├── features/                  [NEW]
│   └── time_domain_features.py
├── models/                    [NEW]
│   ├── xgboost_model.py
│   └── base_dl_model.py
├── orchestrator/
│   ├── preprocessing_orchestrator.py
│   └── ml_orchestrator.py     [NEW]
├── factory/
│   ├── orchestrator_factory.py
│   └── ml_orchestrator_factory.py  [NEW]
└── config/
    └── config.py              [UPDATED: +MLConfig]

main_ml_pipeline.py            [NEW]
ARCHITECTURE.md                [NEW]
requirements.txt               [UPDATED]
```

## Future Extension Points

To add new capabilities, simply implement the interfaces:

**New Feature Extractor**:
```python
class FrequencyFeatureExtractor(IFeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        # FFT, PSD, etc.
```

**New DL Model**:
```python
class LSTMModel(BaseDLModel):
    def train(self, X_train, y_train):
        # LSTM implementation
```

Then wire them in the factory - no changes to orchestrators needed!


# ML Pipeline Fix Summary

## Problem Identified

**Issue 1**: Failure and warning CSVs with same timestamps were being merged during feature extraction
**Issue 2**: Feature extraction was happening AFTER train/test split (incorrect order)

## Solution Implemented

### 1. Feature Extraction with run_id Grouping
**File**: `src/features/time_domain_features.py`

Changed grouping from `["_window"]` to `["_window", "run_id"]` when run_id exists:

```python
group_keys = ["_window", "run_id"] if "run_id" in df.columns else ["_window"]
g = df.groupby(group_keys)[sensor_cols]
```

**Why**: Prevents mixing sensor data from different CSVs with overlapping timestamps.

### 2. Correct Pipeline Order
**Files**: 
- `src/orchestrator/preprocessing_orchestrator.py`
- `src/factory/orchestrator_factory.py`
- `src/orchestrator/ml_orchestrator.py`
- `src/factory/ml_orchestrator_factory.py`
- `main_ml_pipeline.py`

**Old Flow** ❌:
```
Load → Split → Extract Features (train) → Extract Features (test) → Train
```

**New Flow** ✅:
```
Load → Extract Features (all data) → Split → Train
```

**Why**:
- Sliding windows at train/test boundaries would be incomplete
- Statistics computed on smaller subsets
- Risk of data leakage at boundaries
- More efficient (extract once, not twice)

### 3. RunIdSplitter as Default
**File**: `src/factory/orchestrator_factory.py`

Changed default splitter to `RunIdSplitter`:

```python
from src.splitters.data_splitter import RunIdSplitter
splitter = RunIdSplitter()
```

**Why**:
- Each CSV (failure type) properly represented in train/test
- Maintains temporal order within each run_id
- Takes 80% from each CSV for training, 20% for testing
- Prevents bias toward any single failure type

## Benefits

✅ **Data Integrity**: No mixing of different failure types with same timestamps
✅ **Balanced Representation**: Each failure type in both train/test sets
✅ **Temporal Order**: Maintained within each run
✅ **No Data Leakage**: Clean separation between train/test
✅ **Efficient**: Feature extraction done once on all data
✅ **Correct Windowing**: Complete windows without boundary issues

## Configuration

Default settings in `main_ml_pipeline.py`:

```python
preprocessing_orchestrator = OrchestratorFactory.create(
    window_size='10T',           # 10-minute windows
    resample_freq='1T',          # 1-minute resampling
    resample_method='mean',      # Mean aggregation
    extract_features=True        # Extract before splitting
)
```

## Pipeline Flow

1. **Load**: Load all CSVs with run_id assignment
2. **Resample**: Normalize to 1-minute intervals
3. **Validate**: Check data quality
4. **Extract Features**: Group by (window, run_id) → compute statistics
5. **Split**: RunIdSplitter → 80/20 per run_id
6. **Export**: Save train/test CSVs
7. **Train**: XGBoost on extracted features
8. **Evaluate**: Test set metrics

## Files Modified

1. `src/features/time_domain_features.py` - Added run_id grouping
2. `src/orchestrator/preprocessing_orchestrator.py` - Added feature extraction step
3. `src/factory/orchestrator_factory.py` - Added feature extractor, switched to RunIdSplitter
4. `src/orchestrator/ml_orchestrator.py` - Removed duplicate feature extraction
5. `src/factory/ml_orchestrator_factory.py` - Removed feature extractor dependency
6. `main_ml_pipeline.py` - Added extract_features=True parameter


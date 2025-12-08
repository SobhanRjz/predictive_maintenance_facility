# Data Splitting Strategy in Unified Pipeline

## Overview

The unified config-driven pipeline uses run-based splitting to prevent data leakage while maintaining temporal relationships and enabling realistic model evaluation.

## Current Implementation

### Run-Based Splitting (Default)

**Strategy**: Split by `run_id` instead of time windows

```
File 1: normal_3months.csv (run_id=1) → Train
File 2: warning_1_vibration.csv (run_id=2) → Train
File 3: warning_2_temperature.csv (run_id=3) → Test
File 4: failure_1_bearing.csv (run_id=4) → Test
```

### Why Run-Based Splitting?

1. **Prevents Data Leakage**: Each file represents a complete operational run
2. **Maintains Temporal Sequences**: Within each run, time order is preserved
3. **Realistic Evaluation**: Models tested on complete unseen operational scenarios
4. **Handles Imbalanced Data**: Natural class distribution per run

## Configuration

### Layer 1 Config (Full Pipeline)
```yaml
dataset:
  use_timeseries_split: false    # Use run-based splitting
  use_stratification: false      # No additional stratification needed
  test_size: 0.2                # 20% of runs go to test set

  # File lists determine run order
  normal_files: ['normal_3months_30sec_interval-1.csv']      # run_id=1
  failure_files: ['failure_1_bearing_fault.csv', ...]        # run_id=2,3,...
  warning_files: ['warning_1_radial_vibration.csv', ...]     # run_id=...,n
```

### Implementation Details

```python
class RunIdSplitter:
    """Splits data by run_id to prevent leakage between operational runs."""

    def split(self, df, test_size=0.2, random_state=42):
        # Get unique run_ids
        run_ids = sorted(df['run_id'].unique())

        # Split run_ids (not individual rows)
        n_test = max(1, int(len(run_ids) * test_size))
        test_run_ids = run_ids[-n_test:]  # Most recent runs for testing

        # Split dataframe by run_id
        train_mask = ~df['run_id'].isin(test_run_ids)
        test_mask = df['run_id'].isin(test_run_ids)

        return df[train_mask], df[test_mask]
```

## Data Characteristics Analysis

### Your Dataset Properties
- **Time Intervals**: 1-minute intervals (after resampling)
- **Duration**: ~90 days (3 months) per file/run
- **Total Rows**: ~129,600 rows per file
- **Class Distribution** (example per run):
  - Normal: 116,640 (90%)
  - Warning: 10,080 (7.8%)
  - Failure: 2,880 (2.2%)

### Class Imbalance Handling

1. **XGBoost Class Balancing**:
   ```yaml
   hyperparameters:
     scale_pos_weight: [1, 28, 62]  # Balances Normal:Warning:Failure
   ```

2. **Stratified Sampling for Layer 2**:
   ```yaml
   dataset:
     use_stratification: true  # For warning/failure classifiers
   ```

## Alternative Splitting Strategies

### Time-Series Window Splitting (Available)
```yaml
dataset:
  use_timeseries_split: true
  use_stratification: true
  window_size: '10min'  # Rolling window size
```

**Pros**: Fine-grained temporal splitting
**Cons**: Complex implementation, potential data leakage

### Random Splitting (Not Recommended)
```yaml
dataset:
  use_timeseries_split: false
  use_stratification: false
  test_size: 0.2
```

**Issue**: Shuffles time series data, unrealistic evaluation

## Validation Strategy

### Cross-Run Validation
- **Method**: Leave-one-run-out cross-validation
- **Benefit**: Tests generalization across different operational scenarios
- **Implementation**: Rotate which run is held out for testing

### Performance Metrics
- **Primary**: F1-score (handles imbalanced data)
- **Secondary**: Precision, Recall, Accuracy
- **Per-Class**: Individual metrics for Normal/Warning/Failure

## Configuration Examples

### Change Test Set Size
```yaml
dataset:
  test_size: 0.3  # 30% of runs for testing
```

### Enable Stratification (Layer 2)
```yaml
dataset:
  use_stratification: true  # Maintain class distribution in splits
```

### Custom Run Ordering
```yaml
# Control which runs go to train/test by file order
failure_files:
  - 'failure_1_bearing_fault.csv'      # Earlier = more likely train
  - 'failure_2_shaft_misalignment.csv'
  - 'failure_9_impeller_wear.csv'      # Later = more likely test
```

## Quality Assurance

### Leakage Prevention Checks
- **Run ID Overlap**: Verify no run_ids appear in both train/test
- **Temporal Order**: Within runs, time order preserved
- **Feature Consistency**: Same features available in train/test

### Statistical Validation
- **Class Distribution**: Compare train/test class ratios
- **Time Range**: Ensure test set covers appropriate time periods
- **Run Characteristics**: Similar run lengths and patterns

## Performance Impact

### Memory Efficiency
- **Run-based**: Load complete runs into memory
- **Time-based**: Process in streaming windows
- **Current**: Run-based works well for your dataset size

### Training Efficiency
- **Balanced Classes**: XGBoost handles imbalanced data well
- **Early Stopping**: Prevents overfitting on limited data
- **Validation**: Test set represents realistic scenarios

## Troubleshooting

### Data Leakage Detection
```python
# Check for run_id overlap
train_runs = set(train_df['run_id'].unique())
test_runs = set(test_df['run_id'].unique())
overlap = train_runs.intersection(test_runs)

if overlap:
    print(f"WARNING: Data leakage detected: {overlap}")
```

### Class Distribution Issues
- **Problem**: Test set has different class distribution
- **Solution**: Adjust `test_size` or file ordering
- **Prevention**: Use stratified splitting for Layer 2

### Temporal Distribution Issues
- **Problem**: Test set only has recent failures
- **Solution**: Randomize file order or use time-based splitting
- **Prevention**: Ensure representative temporal coverage

## Future Enhancements

### Rolling Window Validation
- **Purpose**: Simulate continuous learning scenario
- **Method**: Expanding window training, rolling test periods
- **Benefit**: More realistic production evaluation

### Cross-Run Validation
- **Implementation**: Automated leave-one-run-out CV
- **Benefit**: Better generalization estimates
- **Config**: `validation: {type: 'cross_run', folds: 5}`

### Online Learning Simulation
- **Purpose**: Test model updates with new data
- **Method**: Incremental training on new runs
- **Benefit**: Production deployment readiness
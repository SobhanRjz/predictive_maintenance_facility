# Time-Series Splitting Strategy Analysis

## Your Data Characteristics

Based on analysis of your datasets:

### Dataset Properties:
- **Time Intervals**: 1-minute intervals consistently
- **Duration**: ~90 days (3 months) per file
- **Total Rows**: 129,600 rows per file
- **Class Distribution** (e.g., failure_1_bearing_fault):
  - Normal: 116,640 (90%)
  - Warning: 10,080 (7.8%)
  - Failure: 2,880 (2.2%)

### Class Imbalance Issue:
- **Normal**: ~90% of data
- **Warning**: ~8% of data  
- **Failure**: ~2% of data (highly imbalanced)

## Your Stratified Time-Series Splitter - EVALUATION

### ✅ STRENGTHS:

1. **Temporal Order Preserved**: Test set is contiguous suffix in time - CORRECT
2. **No Data Leakage**: Entire windows go to train/test - GOOD
3. **Class-Aware**: Tries to maintain class distribution - IMPORTANT for your imbalanced data
4. **Window-Based**: Works on time windows, not individual rows - APPROPRIATE

### ⚠️ POTENTIAL ISSUES:

1. **Window Size Selection** (`1H` default):
   - Your data: 1-minute intervals
   - 1 hour window = 60 samples per window
   - With 2.2% failure rate, many windows may have 0 failures
   - **RECOMMENDATION**: Use smaller windows like `'5T'` or `'10T'` (5-10 minutes)

2. **Backwards Accumulation**:
   - Good: Uses most recent data for testing (realistic for production)
   - Risk: If failures cluster at end, may over-represent in test set
   - For your data: This is actually **GOOD** since failures typically occur after warnings

3. **Stopping Condition**:
   - Stops when target counts met **AND** minimum rows reached
   - Risk: May take too many windows if failure class is sparse
   - **SOLUTION**: Already handled with `max(1, int(cnt * test_size))`

## RECOMMENDATION

### ✅ YES, Use StratifiedTimeSeriesSplitter for your data

**Reasons:**
1. You have **severe class imbalance** (90% normal, 2% failure)
2. Time-series nature requires temporal ordering
3. Motor failures typically follow a pattern: Normal → Warning → Failure
4. Testing on recent data simulates real-world deployment

### Suggested Configuration:

```python
# For 1-minute interval data with imbalance
splitter = StratifiedTimeSeriesSplitter(
    timestamp_col="timestamp",
    stratify_col="health_status",
    window_size="10T"  # 10 minutes = 10 samples per window
)
```

**Why 10T?**
- 10 minutes = 10 rows per window
- With 2.2% failure rate: ~0.22 failures per window on average
- Balances granularity vs. having enough samples per window

### Alternative Approaches:

1. **Simple TimeSeriesSplitter** (if you want pure chronological split):
   - Simpler, faster
   - Risk: Test set may have different class distribution than train

2. **StratifiedTimeSeriesSplitter with adaptive window**:
   - Use `'5T'` for more granular stratification
   - Use `'30T'` for coarser but more stable stratification

## Final Answer

**Your implementation is GOOD and APPROPRIATE for time-series classification with class imbalance.**

Minor adjustments:
- Reduce window_size from '1H' to '10T' or '5T' 
- Consider the temporal pattern of failures in your specific motor data

The approach correctly balances:
✅ Temporal ordering (no future leakage)
✅ Class distribution (handles imbalance)
✅ Realistic evaluation (tests on recent data)


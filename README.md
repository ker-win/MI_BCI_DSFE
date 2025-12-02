# BCI Motor Imagery Classification

## Configuration

### Classification Type
You can adjust the classification task type in `load_data.py` within the `Args` class.

```python
# load_data.py

class Args:
    # ...
    # 選擇分類任務類型: '4_class' (左/右/腳/舌) 或 '2_class' (左/右)
    classification_type = '4_class' 
    # classification_type = '2_class' 
    # ...
```

- **`4_class`**: Classifies 4 motor imagery tasks: Left Hand, Right Hand, Foot, Tongue.
- **`2_class`**: Classifies 2 motor imagery tasks: Left Hand vs Right Hand.

## Change Log

### 2025-12-02
- **Fixed Data Slicing Bug**: Modified `dsfe/preprocess.py` to remove redundant data slicing. Previously, the code attempted to re-slice the data using `EPOCH_TMIN` and `EPOCH_TMAX` as offsets, even though the data loaded by `load_data.py` was already sliced to this window. This caused the data to be truncated (e.g., from 750 samples to 500 samples), leading to `ValueError: Invalid window indices` during feature extraction when accessing later time windows.
- **Enhanced Multi-Window Support**: Updated `dsfe/train_eval.py` to support Feature-Dependent Correlation Coefficient (FDCC) band selection within the Multi-Window framework. Now, if both `USE_MULTI_WINDOW` and `USE_FDCC` are enabled, the optimal frequency band is selected independently for each time window.
- **Added Power Time Course (PTC) Features**: Implemented a new feature extraction method `compute_ptc_features` in `dsfe/features.py`. This method extracts the power time course of specific frequency bands (default: Mu 8-13Hz, Beta 13-30Hz) by calculating the average power in sliding windows (default: 0.2s). Added configuration flags `USE_PTC`, `PTC_BANDS`, `PTC_WINDOW_SIZE`, and `PTC_OVERLAP` to `dsfe/config.py`.
- **Implemented Global ReliefF Fusion**: Added `USE_GLOBAL_FUSION` mode in `dsfe/config.py`. When enabled, all active feature sets (FTA, RG, PTC) are Z-score normalized and concatenated into a single global feature vector before applying ReliefF selection. This ensures fair competition between features of different scales.
- **Added Feature Contribution Analysis**: The Global Fusion mode now calculates and prints the percentage of selected features contributed by each feature type (e.g., "FTA: 40%, RG: 30%, PTC: 30%"), allowing for direct assessment of feature importance.
- **Implemented Phase 1 Analysis (Single FTS Wrapper)**: Created `run_phase1_single_fts.py` and `dsfe/fgsft.py` to implement the first phase of the FGSFT framework adaptation. This script performs a fine-grained analysis by segmenting the time-frequency space into small "FTS" blocks (e.g., 0.5s window x 4Hz band), evaluating each block individually using a wrapper-based approach (Linear SVM with Inner CV), and selecting the single best block for classification. This helps identify the most discriminative time-frequency locations for each subject.
- **Added Multi-Processing Support**: Updated `run_phase1_single_fts.py` to support multi-core processing using `joblib`. Added `USE_MULTIPROCESSING` and `N_JOBS` configuration variables to allow users to toggle parallel execution and specify the number of cores (default: 4), significantly reducing computation time.

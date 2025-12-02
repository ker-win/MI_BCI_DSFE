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

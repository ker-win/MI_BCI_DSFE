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

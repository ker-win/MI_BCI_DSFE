import numpy as np
import sys
import os
from . import config

# Add parent directory to path to import load_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from load_data import MotorImageryDataset, Args
except ImportError:
    # Fallback if load_data is not found
    print("Error: Could not import load_data.py. Please ensure it is in the parent directory.")
    pass

def load_subject_data(subject_id, channels=None):
    """
    Load data for a specific subject using the existing MotorImageryDataset class.
    
    Args:
        subject_id (str or int): Subject ID (e.g., '1', 1).
        channels (list of int, optional): Not used with current load_data.py implementation.
    
    Returns:
        X (np.ndarray): Shape (n_trials, n_channels, n_times)
        y (np.ndarray): Shape (n_trials,) - Integer Labels
        fs (int): Sampling rate
    """
    # Handle subject_id formatting
    if isinstance(subject_id, int):
        subject_id = str(subject_id)
    
    # Use Args.data_path from load_data.py
    data_path = Args.data_path
    
    # Initialize dataset loader with correct signature: (file_path, subject_id)
    dataset = MotorImageryDataset(data_path, subject_id)
    
    # Get epochs using config parameters
    # Note: dataset.get_epochs returns (trials, labels)
    # trials: (n_trials, n_channels, n_times)
    # labels: array of strings ('left', 'right', etc.)
    X_raw, y_raw_labels = dataset.get_epochs(config.EPOCH_TMIN, config.EPOCH_TMAX)
    
    if len(X_raw) == 0:
        raise ValueError(f"No data found for subject {subject_id}")

    # Filter classes based on Args.class_type
    # Args.class_type is a list like ['left', 'right']
    mask = np.isin(y_raw_labels, Args.class_type)
    X = X_raw[mask]
    y_labels = y_raw_labels[mask]
    
    # Convert string labels to integers using Args.label_dict
    # Args.label_dict is like {'left': 0, 'right': 1}
    y = np.array([Args.label_dict[label] for label in y_labels])
    
    return X, y, dataset.fs

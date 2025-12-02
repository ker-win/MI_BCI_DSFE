import numpy as np
import mne
from . import config

def preprocess_data(X, y, fs):
    """
    Apply preprocessing steps:
    1. Average Reference
    2. Epoching (slicing to desired time window)
    3. Lowpass Filtering
    
    Args:
        X (np.ndarray): (n_trials, n_channels, n_times)
        y (np.ndarray): (n_trials,)
        fs (int): Sampling rate
        
    Returns:
        X_processed (np.ndarray): Preprocessed data
        y (np.ndarray): Labels (unchanged)
    """
    # 1. Average Reference
    # X is (n_trials, n_channels, n_times)
    # We subtract the mean across channels for each time point
    X_avg = X - np.mean(X, axis=1, keepdims=True)
    
    # 2. Epoching / Slicing
    # The data loaded by io.load_subject_data (via MotorImageryDataset.get_epochs)
    # is ALREADY sliced from config.EPOCH_TMIN to config.EPOCH_TMAX.
    # So X starts at EPOCH_TMIN, not 0.
    # We do not need to slice again using TMIN as an offset.
    
    X_sliced = X_avg
    
    # 3. Lowpass Filtering
    # We can use MNE for filtering or scipy.
    # Using MNE's filter_data for convenience and robustness
    
    # mne.filter.filter_data expects (n_epochs, n_channels, n_times) or (n_channels, n_times)
    # It handles the array directly.
    
    X_filtered = mne.filter.filter_data(
        X_sliced.astype(np.float64), 
        sfreq=fs, 
        l_freq=None, 
        h_freq=config.LOWPASS, 
        method='iir',  # fast
        verbose=False
    )
    
    return X_filtered, y

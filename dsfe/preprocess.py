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
    # The raw data loaded might be longer than we need.
    # config.EPOCH_TMIN and config.EPOCH_TMAX define the window relative to cue.
    # Assuming the loaded X starts at cue (0s) or we need to adjust.
    # Based on load_data.py, it extracts from 'start' to 'stop'.
    # 'start' is events_position.
    # We need to ensure we slice exactly [tmin, tmax].
    
    # Calculate samples
    n_samples = X.shape[2]
    t_start = 0.0 # Assuming X starts at 0
    
    # If we want 0 to 0.85s
    idx_start = int(config.EPOCH_TMIN * fs)
    idx_end = int(config.EPOCH_TMAX * fs)
    
    if idx_end > n_samples:
        print(f"Warning: Desired epoch length ({idx_end}) exceeds data length ({n_samples}). Using available length.")
        idx_end = n_samples
        
    X_sliced = X_avg[:, :, idx_start:idx_end]
    
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

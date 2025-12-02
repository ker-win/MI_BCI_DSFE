import numpy as np
import mne
from . import features

def generate_time_windows(epoch_tmin, epoch_tmax, window_sizes, overlap_ratio=0.5):
    """
    Generate a list of time windows (start, end) within the epoch.
    
    Args:
        epoch_tmin (float): Start of the analysis epoch (e.g., 0.5).
        epoch_tmax (float): End of the analysis epoch (e.g., 4.0).
        window_sizes (list of float): List of window lengths (e.g., [0.5, 1.0]).
        overlap_ratio (float): Overlap ratio (0.0 to 1.0).
        
    Returns:
        windows (list of tuple): List of (t_start, t_end).
    """
    windows = []
    duration = epoch_tmax - epoch_tmin
    
    for length in window_sizes:
        step = length * (1 - overlap_ratio)
        if step <= 0:
            continue
            
        # Generate windows
        # t_start relative to epoch_tmin
        current_start = 0.0
        while current_start + length <= duration + 1e-5: # Small epsilon for float errors
            t_start = epoch_tmin + current_start
            t_end = t_start + length
            windows.append((t_start, t_end))
            current_start += step
            
    return windows

def generate_freq_bands(f_min, f_max, width, step):
    """
    Generate a list of frequency bands (f_low, f_high).
    
    Args:
        f_min (float): Min frequency.
        f_max (float): Max frequency.
        width (float): Bandwidth.
        step (float): Step size.
        
    Returns:
        bands (list of tuple): List of (f_low, f_high).
    """
    bands = []
    current_f = f_min
    while current_f + width <= f_max + 1e-5:
        bands.append((current_f, current_f + width))
        current_f += step
    return bands

def extract_fts_features_from_epoch(X_epoch, fs, t_slice_indices, freq_band, ptc_params=None, X_filtered=None):
    """
    Extract features for a specific FTS.
    
    Args:
        X_epoch (np.ndarray): (n_trials, n_channels, n_times) - The full analysis epoch (broadband).
        fs (float): Sampling rate.
        t_slice_indices (tuple): (start_idx, end_idx) indices to slice X_epoch.
        freq_band (tuple): (f_low, f_high).
        ptc_params (dict): {'window_size': float, 'overlap': float}
        X_filtered (np.ndarray, optional): Pre-filtered full epoch data. If None, filters X_epoch.
        
    Returns:
        Z (np.ndarray): (n_trials, n_features) - Concatenated FTA and PTC features.
    """
    start_idx, end_idx = t_slice_indices
    f1, f2 = freq_band
    
    # --- 1. FTA Features ---
    # For FTA (FFT), we slice the raw data first (windowing) then compute FFT.
    # This is standard STFT approach.
    X_time_raw = X_epoch[:, :, start_idx:end_idx]
    fta_feats = features.compute_fta_features(X_time_raw, fs, freq_band)
    
    # --- 2. PTC Features ---
    # For PTC (Power Time Course), we must filter the FULL epoch first to avoid edge artifacts.
    # Then slice the filtered data.
    
    if X_filtered is not None:
        # Use pre-computed filtered data
        X_filt_full = X_filtered
    else:
        # Filter full epoch (expensive if repeated)
        X_filt_full = mne.filter.filter_data(
            X_epoch.astype(np.float64), 
            sfreq=fs, 
            l_freq=f1, 
            h_freq=f2, 
            method='iir', 
            verbose=False
        )
    
    # Slice filtered data
    X_filt_slice = X_filt_full[:, :, start_idx:end_idx]
    
    # Compute Power (squared signal)
    X_pow = X_filt_slice ** 2
    
    # Compute Sliding Window Average on the slice
    if ptc_params is None:
        ptc_params = {'window_size': 0.2, 'overlap': 0.0}
        
    window_size = ptc_params['window_size']
    overlap = ptc_params['overlap']
    
    n_samples_window = int(window_size * fs)
    n_samples_overlap = int(overlap * fs)
    step = n_samples_window - n_samples_overlap
    
    if step <= 0:
        step = n_samples_window # Fallback
        
    n_times_slice = X_filt_slice.shape[2]
    band_window_powers = []
    
    # Iterate through windows within the FTS slice
    for w_start in range(0, n_times_slice - n_samples_window + 1, step):
        w_end = w_start + n_samples_window
        win_pow = np.mean(X_pow[:, :, w_start:w_end], axis=2)
        band_window_powers.append(win_pow)
        
    if not band_window_powers:
        # If FTS slice is shorter than PTC window, take mean of whole slice
        if n_times_slice > 0:
            win_pow = np.mean(X_pow, axis=2)
            band_window_powers.append(win_pow)
        else:
            ptc_feats = np.zeros((X_epoch.shape[0], 0))
            
    if band_window_powers:
        W = np.stack(band_window_powers, axis=1)
        ptc_feats = W.reshape(X_epoch.shape[0], -1)
    
    # --- 3. Concatenate ---
    Z = np.concatenate([fta_feats, ptc_feats], axis=1)
    
    return Z

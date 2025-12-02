import numpy as np
from scipy.linalg import fractional_matrix_power, logm
import mne

# Try importing pyriemann, if not available, we might need a fallback or error
try:
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_riemann
except ImportError:
    print("Warning: pyriemann not found. RG features will fail.")

def compute_fta_features(X, fs, band):
    """
    Compute Fourier Transform Amplitude (FTA) features.
    
    Args:
        X (np.ndarray): (n_trials, n_channels, n_times)
        fs (float): Sampling rate
        band (tuple): (f1, f2) Frequency band
        
    Returns:
        F (np.ndarray): (n_trials, n_features)
    """
    f1, f2 = band
    n_trials, n_channels, n_times = X.shape
    
    # FFT frequencies
    freqs = np.fft.rfftfreq(n_times, d=1.0/fs)
    
    # Indices within the band
    idx_band = np.where((freqs >= f1) & (freqs <= f2))[0]
    
    if len(idx_band) == 0:
        # Fallback if band is too narrow or out of range
        return np.zeros((n_trials, 0))

    F_list = []
    for trial in range(n_trials):
        # FFT per channel
        # rfft returns complex values
        fft_vals = np.fft.rfft(X[trial], axis=-1)
        
        # Amplitude
        amp = np.abs(fft_vals)[:, idx_band]
        
        # Flatten to 1D vector
        F_list.append(amp.reshape(-1))
        
    F = np.stack(F_list, axis=0)
    return F

def compute_rg_features(X, fs, band, training_data=None):
    """
    Compute Riemannian Geometry (RG) features.
    
    Args:
        X (np.ndarray): (n_trials, n_channels, n_times)
        fs (float): Sampling rate
        band (tuple): (f1, f2) Frequency band. 
                      Note: X should ideally be bandpassed to this band BEFORE calling this,
                      or we do it here. The paper implies bandpass first.
        training_data (dict, optional): Contains 'P_G' (Riemannian mean) computed from training set.
                                        If None, computes P_G from X (assuming X is training set).
                                        If provided, maps X to tangent space using provided P_G.
        
    Returns:
        G (np.ndarray): (n_trials, n_features)
        meta (dict): Metadata containing P_G if computed.
    """
    f1, f2 = band
    
    # Bandpass filter X to [f1, f2]
    # We use MNE filter
    X_band = mne.filter.filter_data(
        X.astype(np.float64), 
        sfreq=fs, 
        l_freq=f1, 
        h_freq=f2, 
        method='iir', 
        verbose=False
    )
    
    # Compute Covariance Matrices
    # pyriemann covariances: (n_matrices, n_channels, n_channels)
    covs = covariances(X_band, estimator='oas') # 'oas' or 'lw' or 'scm'
    
    # Riemannian Mean
    if training_data and 'P_G' in training_data:
        P_G = training_data['P_G']
    else:
        # Compute mean from current data (Training phase)
        P_G = mean_riemann(covs)
        
    # Map to Tangent Space
    # s = upper(log(P_G^-1/2 * P * P_G^-1/2))
    
    # P_G^-1/2
    P_G_inv_sqrt = fractional_matrix_power(P_G, -0.5)
    
    feats = []
    for P in covs:
        # Map to tangent space
        T = P_G_inv_sqrt @ P @ P_G_inv_sqrt
        L = logm(T)
        
        # Vectorize (upper triangle)
        # Note: The paper mentions multiplying off-diagonals by sqrt(2)
        upper_inds = np.triu_indices_from(L)
        upper = L[upper_inds]
        
        # Apply sqrt(2) to off-diagonal elements
        # We can identify off-diagonals by checking if row != col in upper_inds
        rows, cols = upper_inds
        off_diag_mask = rows != cols
        upper[off_diag_mask] *= np.sqrt(2)
        
        feats.append(upper)
        
    G = np.stack(feats, axis=0)
    
    return G, {'P_G': P_G}

def compute_ptc_features(X, fs, bands, window_size, overlap=0.0):
    """
    Compute Power Time Course (PTC) features.
    
    For each band in bands:
        1. Bandpass filter X.
        2. Compute instantaneous power (squared signal).
        3. Average power in sliding windows.
        
    Args:
        X (np.ndarray): (n_trials, n_channels, n_times)
        fs (float): Sampling rate
        bands (dict): Dictionary of bands, e.g., {'mu': (8, 13), 'beta': (13, 30)}
        window_size (float): Window size in seconds
        overlap (float): Overlap in seconds
        
    Returns:
        F (np.ndarray): (n_trials, n_features)
    """
    n_trials, n_channels, n_times = X.shape
    
    n_samples_window = int(window_size * fs)
    n_samples_overlap = int(overlap * fs)
    step = n_samples_window - n_samples_overlap
    
    if step <= 0:
        raise ValueError("Overlap must be smaller than window size.")
        
    all_band_features = []
    
    for band_name, (f1, f2) in bands.items():
        # 1. Filter
        X_filt = mne.filter.filter_data(
            X.astype(np.float64), 
            sfreq=fs, 
            l_freq=f1, 
            h_freq=f2, 
            method='iir', 
            verbose=False
        )
        
        # 2. Power (squared signal)
        # Alternatively, could use Hilbert envelope: np.abs(hilbert(X_filt))**2
        # But simple squared signal is common for "power".
        X_pow = X_filt ** 2
        
        # 3. Sliding Window Average
        band_window_powers = []
        
        # Iterate through windows
        # Range: 0 to n_times - n_samples_window, step=step
        for start_idx in range(0, n_times - n_samples_window + 1, step):
            end_idx = start_idx + n_samples_window
            
            # Average power in this window: (n_trials, n_channels)
            win_pow = np.mean(X_pow[:, :, start_idx:end_idx], axis=2)
            band_window_powers.append(win_pow)
            
        if not band_window_powers:
             # Handle case where data is shorter than one window
             print(f"Warning: Data length ({n_times}) is shorter than PTC window ({n_samples_window}).")
             continue

        # Concatenate windows for this band: (n_trials, n_channels * n_windows)
        # band_window_powers is list of (n_trials, n_channels)
        # Stack along axis 1 (channels) -> (n_trials, n_channels, n_windows) -> flatten
        # Or just concatenate all features.
        # User said: P_mu(t1), ..., P_mu(tN), P_beta(t1), ...
        # So we want [Channel1_Win1, Channel1_Win2..., Channel2_Win1...] or [Win1_Ch1, Win1_Ch2...]
        # Let's stack windows first: (n_trials, n_windows, n_channels)
        # Then flatten to (n_trials, n_features)
        
        # Stack windows: (n_trials, n_windows, n_channels)
        # But wait, band_window_powers is list of (n_trials, n_channels)
        # np.stack(..., axis=1) -> (n_trials, n_windows, n_channels)
        W = np.stack(band_window_powers, axis=1) 
        
        # Flatten: (n_trials, n_windows * n_channels)
        W_flat = W.reshape(n_trials, -1)
        
        all_band_features.append(W_flat)
        
    if not all_band_features:
        return np.zeros((n_trials, 0))
        
    # Concatenate bands: (n_trials, total_features)
    F = np.concatenate(all_band_features, axis=1)
    
    return F

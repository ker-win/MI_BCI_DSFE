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

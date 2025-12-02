# DSFE Configuration

# --- Ablation Study Flags ---
# Set these to True/False to enable/disable specific components
# USE_FTA = True          # Use Fourier Transform Amplitudes features
# USE_RG = True           # Use Riemannian Geometry features
# USE_FDCC = True         # Use Feature-Dependent Correlation Coefficient band selection
# USE_RELIEFF = True      # Use ReliefF feature selection/fusion
# USE_ENSEMBLE = True     # Use Ensemble Learning (SVM+RF+NB)

# --- Ablation Study Flags ---
USE_FTA = True          # Use Fourier Transform Amplitudes features
USE_RG = False           # Use Riemannian Geometry features
USE_FDCC = True         # Use Feature-Dependent Correlation Coefficient band selection
USE_RELIEFF = True      # Use ReliefF feature selection/fusion
USE_ENSEMBLE = True     # Use Ensemble Learning (SVM+RF+NB)
USE_PTC = True          # Use Power Time Course features

# --- PTC Settings ---
PTC_BANDS = {'mu': (8.0, 13.0), 'beta': (13.0, 30.0)}
PTC_WINDOW_SIZE = 0.2   # Window size in seconds
PTC_OVERLAP = 0.0       # Overlap in seconds

# --- Multi-Window Settings ---
USE_MULTI_WINDOW = True
TIME_WINDOWS = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)] # Time windows relative to cue (s)

# If USE_FDCC is False, we need a default band
DEFAULT_BAND = (8.0, 30.0)

# --- General Settings ---
FREQ_RANGE = (0.0, 30.0)   # Search range for FDCC
SUBBAND_WIDTH = 2.0        # Width of subbands for FDCC
FS_TARGET = 250            # Target sampling rate (Hz) - User specified 250Hz
EPOCH_TMIN = 1.0           # Start of epoch relative to cue (s)
EPOCH_TMAX = 4.0          # End of epoch relative to cue (s) (Paper uses 0-0.85s)
LOWPASS = 30.0             # Lowpass filter cutoff (Hz)

# --- FDCC Settings ---
T_CANDIDATES = [3, 4, 5, 6, 7, 8]   # Number of top subbands to consider
N_FOLDS_FDCC = 5           # Folds for internal CV in FDCC

# --- ReliefF Settings ---
RELIEFF_KEEP_RATIO = 0.25  # Ratio of features to keep
N_NEIGHBORS_RELIEFF = 20   # k nearest neighbors
N_ITER_RELIEFF = 500       # Number of iterations (if using iterative implementation)

# --- Ensemble Settings ---
SVM_WEIGHT = 0.16
RF_WEIGHT  = 0.09
NB_WEIGHT  = 0.08
N_TREES_RF = 100

# --- Evaluation Settings ---
N_FOLDS_EVAL = 10          # Folds for outer CV
RANDOM_STATE = 42

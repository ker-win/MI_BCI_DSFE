import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from . import features
from . import config

def get_subbands(freq_range, width):
    f_start, f_end = freq_range
    subbands = []
    curr = f_start
    while curr + width <= f_end:
        subbands.append((curr, curr + width))
        curr += width # Non-overlapping? Paper says "step size of 2 Hz" and "width of 2 Hz" -> non-overlapping if step=width
        # If step < width, they overlap. Paper: "decomposed into sub-bands with a bandwidth of 2 Hz... in the range 0-30 Hz"
        # Usually implies 0-2, 2-4...
    return subbands

def corr_features_labels(F, y):
    """
    Calculate Pearson correlation coefficient between each feature and labels.
    Since labels are categorical, we might need to handle this.
    The paper says: "correlation coefficient r between the feature vector and the class labels".
    This implies labels are treated as numeric or one-vs-rest?
    "The class labels were 1, 2, 3, 4...".
    Pearson correlation works for ordinal/continuous. For categorical, it's a bit heuristic but common in BCI papers.
    Alternatively, they might mean Point-Biserial if binary, or ANOVA F-value.
    Paper text: "calculate the correlation coefficients... absolute values |r|... average |r|..."
    We will assume simple Pearson with integer labels as per paper description.
    """
    n_features = F.shape[1]
    corrs = []
    for i in range(n_features):
        f_vec = F[:, i]
        # Handle constant features (std=0) to avoid NaN
        if np.std(f_vec) == 0:
            corrs.append(0.0)
        else:
            r = np.corrcoef(f_vec, y)[0, 1]
            corrs.append(r)
    return np.array(corrs)

def merge_adjacent_bands(bands):
    """
    Merge adjacent bands.
    bands: list of (f1, f2) tuples, sorted by frequency?
    The input 'bands' is "top T subbands". They might not be adjacent.
    Paper: "combine the adjacent sub-bands... e.g. 0-2, 2-4 -> 0-4"
    """
    # First, sort bands by start freq
    sorted_bands = sorted(bands, key=lambda x: x[0])
    
    merged = []
    if not sorted_bands:
        return merged
        
    curr_start, curr_end = sorted_bands[0]
    
    for i in range(1, len(sorted_bands)):
        next_start, next_end = sorted_bands[i]
        if np.isclose(curr_end, next_start):
            # Adjacent, merge
            curr_end = next_end
        else:
            # Not adjacent, push current and start new
            merged.append((curr_start, curr_end))
            curr_start, curr_end = next_start, next_end
            
    merged.append((curr_start, curr_end))
    return merged

def fdcc_select_band(X, y, fs, feature_type):
    """
    Select the best frequency band using FDCC.
    
    Args:
        X: (n_trials, n_channels, n_times)
        y: (n_trials,)
        fs: sampling rate
        feature_type: 'fta' or 'rg'
        
    Returns:
        best_band: (f1, f2)
    """
    subbands = get_subbands(config.FREQ_RANGE, config.SUBBAND_WIDTH)
    
    # 1. Calculate score for each subband
    subband_scores = []
    
    # Pre-calculate features for all subbands to save time?
    # Or calculate on the fly.
    
    for band in subbands:
        if feature_type == 'fta':
            F = features.compute_fta_features(X, fs, band)
            # FTA returns (n_trials, n_features)
        elif feature_type == 'rg':
            F, _ = features.compute_rg_features(X, fs, band)
            
        r = corr_features_labels(F, y)
        score = np.mean(np.abs(r)) if len(r) > 0 else 0
        subband_scores.append((score, band))
        
    # Sort by score descending
    subband_scores.sort(key=lambda x: x[0], reverse=True)
    sorted_subbands = [x[1] for x in subband_scores]
    
    # 2. Cross-validation to select best T
    best_cv_acc = -1.0
    best_band_overall = config.DEFAULT_BAND # Fallback
    
    # If T candidates are more than available subbands, clip
    t_candidates = [t for t in config.T_CANDIDATES if t <= len(subbands)]
    
    for T in t_candidates:
        # Top T subbands
        top_T = sorted_subbands[:T]
        
        # Merge adjacent
        candidate_bands = merge_adjacent_bands(top_T)
        
        # Select best candidate band based on avg |r| on WHOLE training set
        # (Paper says: "calculate the correlation... select the band with maximum average correlation")
        best_cand_score = -1.0
        best_cand_band = None
        
        for cand in candidate_bands:
            if feature_type == 'fta':
                F = features.compute_fta_features(X, fs, cand)
            elif feature_type == 'rg':
                F, _ = features.compute_rg_features(X, fs, cand)
                
            r = corr_features_labels(F, y)
            score = np.mean(np.abs(r)) if len(r) > 0 else 0
            
            if score > best_cand_score:
                best_cand_score = score
                best_cand_band = cand
        
        if best_cand_band is None:
            continue
            
        # Evaluate this T (and its best band) using CV
        # Paper: "The parameter T is determined by cross-validation"
        
        if feature_type == 'fta':
            F_eval = features.compute_fta_features(X, fs, best_cand_band)
        elif feature_type == 'rg':
            F_eval, _ = features.compute_rg_features(X, fs, best_cand_band)
            
        cv = StratifiedKFold(n_splits=config.N_FOLDS_FDCC, shuffle=True, random_state=config.RANDOM_STATE)
        accs = []
        for train_idx, val_idx in cv.split(F_eval, y):
            clf = SVC(kernel='rbf') # Simple classifier for selection
            clf.fit(F_eval[train_idx], y[train_idx])
            acc = clf.score(F_eval[val_idx], y[val_idx])
            accs.append(acc)
            
        mean_acc = np.mean(accs)
        
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_band_overall = best_cand_band
            
    return best_band_overall

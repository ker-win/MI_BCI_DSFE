import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import mne

# Add current directory to path to import dsfe
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dsfe import io, fgsft, config

def evaluate_fts_candidate(params, X_train, y_train, fs, ptc_params, n_inner_folds, random_state, X_filtered=None):
    """
    Helper function to evaluate a single FTS candidate.
    Must be at module level for joblib pickling.
    """
    # 1. Extract Features for ALL train data
    # Pass pre-filtered data if available
    Z_train_full = fgsft.extract_fts_features_from_epoch(
        X_train, fs, params['t_indices'], params['band'], ptc_params, X_filtered=X_filtered
    )
    
    # 2. Inner CV
    skf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    inner_accs = []
    
    for in_train_idx, in_val_idx in skf_inner.split(Z_train_full, y_train):
        Z_in_train, Z_in_val = Z_train_full[in_train_idx], Z_train_full[in_val_idx]
        y_in_train, y_in_val = y_train[in_train_idx], y_train[in_val_idx]
        
        # Normalize
        scaler = StandardScaler()
        Z_in_train_scaled = scaler.fit_transform(Z_in_train)
        Z_in_val_scaled = scaler.transform(Z_in_val)
        
        # Train Linear SVM
        # Increased max_iter to 10000 to prevent convergence warnings
        clf = LinearSVC(dual='auto', random_state=random_state, max_iter=10000)
        clf.fit(Z_in_train_scaled, y_in_train)
        
        # Evaluate
        y_pred = clf.predict(Z_in_val_scaled)
        inner_accs.append(accuracy_score(y_in_val, y_pred))
    
    avg_inner_acc = np.mean(inner_accs)
    return avg_inner_acc, params

def run_phase1_analysis(subject_ids):
    """
    Run Phase 1 Analysis: Single FTS Wrapper Selection.
    """
    
    # --- Configuration ---
    # Analysis Epoch
    EPOCH_START = 0.5
    EPOCH_END = 4.0
    
    # Grid Settings
    WINDOW_SIZES = [0.5, 1.0, 1.5]
    WINDOW_OVERLAP = 0.5
    
    FREQ_MIN = 4.0
    FREQ_MAX = 40.0
    FREQ_WIDTH = 4.0
    FREQ_STEP = 2.0
    
    # PTC Settings (for inner FTS)
    PTC_PARAMS = {'window_size': 0.2, 'overlap': 0.0}
    
    # CV Settings
    N_OUTER_FOLDS = 5
    N_INNER_FOLDS = 5
    RANDOM_STATE = 42
    
    # Parallel Settings
    USE_MULTIPROCESSING = True
    N_JOBS = 4  # Number of cores to use
    
    results = []
    
    for subject_id in subject_ids:
        print(f"\nProcessing Subject {subject_id}...")
        
        # 1. Load Data
        original_tmin = config.EPOCH_TMIN
        original_tmax = config.EPOCH_TMAX
        config.EPOCH_TMIN = EPOCH_START
        config.EPOCH_TMAX = EPOCH_END
        
        try:
            X_full, y, fs = io.load_subject_data(subject_id)
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
            config.EPOCH_TMIN = original_tmin
            config.EPOCH_TMAX = original_tmax
            continue
            
        config.EPOCH_TMIN = original_tmin
        config.EPOCH_TMAX = original_tmax
        
        print(f"Data shape: {X_full.shape}, Labels: {y.shape}")
        
        # 2. Generate Grid
        duration = EPOCH_END - EPOCH_START
        time_windows_rel = fgsft.generate_time_windows(0.0, duration, WINDOW_SIZES, WINDOW_OVERLAP)
        freq_bands = fgsft.generate_freq_bands(FREQ_MIN, FREQ_MAX, FREQ_WIDTH, FREQ_STEP)
        
        print(f"Generated {len(time_windows_rel)} time windows and {len(freq_bands)} freq bands.")
        print(f"Total FTS candidates: {len(time_windows_rel) * len(freq_bands)}")
        
        # Pre-calculate indices for time windows
        time_indices = []
        for t_start, t_end in time_windows_rel:
            idx_start = int(t_start * fs)
            idx_end = int(t_end * fs)
            time_indices.append((idx_start, idx_end))
            
        # 3. Outer CV
        skf_outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        outer_fold_accs = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf_outer.split(X_full, y)):
            print(f"  Outer Fold {fold_idx+1}/{N_OUTER_FOLDS}")
            
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # --- Optimization: Pre-compute Filtered Data (Caching) ---
            print("    Pre-computing filtered data for all bands...")
            filtered_cache = {}
            for band in tqdm(freq_bands, desc="    Filtering", leave=False):
                f1, f2 = band
                # Filter X_train once for this band
                X_filt = mne.filter.filter_data(
                    X_train.astype(np.float64), 
                    sfreq=fs, 
                    l_freq=f1, 
                    h_freq=f2, 
                    method='iir', 
                    verbose=False
                )
                filtered_cache[band] = X_filt
            
            # --- FTS Wrapper Selection (on X_train) ---
            
            # Create a list of all FTS parameters to iterate
            all_fts_params = []
            for t_idx, (t_start_idx, t_end_idx) in enumerate(time_indices):
                for f_idx, band in enumerate(freq_bands):
                    all_fts_params.append({
                        't_indices': (t_start_idx, t_end_idx),
                        'band': band,
                        't_window_rel': time_windows_rel[t_idx],
                        'id': f"{t_idx}_{f_idx}"
                    })
            
            best_fts_score = -1.0
            best_fts_params = None
            
            if USE_MULTIPROCESSING:
                # Parallel Execution
                print(f"    Evaluating {len(all_fts_params)} FTS candidates using {N_JOBS} cores...")
                
                parallel_results = Parallel(n_jobs=N_JOBS)(
                    delayed(evaluate_fts_candidate)(
                        params, X_train, y_train, fs, PTC_PARAMS, N_INNER_FOLDS, RANDOM_STATE,
                        X_filtered=filtered_cache[params['band']] # Pass specific filtered data
                    ) for params in tqdm(all_fts_params, desc="    Parallel FTS Eval", leave=False)
                )
                
                # Find best
                for score, params in parallel_results:
                    if score > best_fts_score:
                        best_fts_score = score
                        best_fts_params = params
            else:
                # Sequential Execution
                for params in tqdm(all_fts_params, desc="    Sequential FTS Eval", leave=False):
                    score, _ = evaluate_fts_candidate(
                        params, X_train, y_train, fs, PTC_PARAMS, N_INNER_FOLDS, RANDOM_STATE,
                        X_filtered=filtered_cache[params['band']]
                    )
                    if score > best_fts_score:
                        best_fts_score = score
                        best_fts_params = params
            
            # --- End Wrapper Selection ---
            
            print(f"    Best FTS: Time={best_fts_params['t_window_rel']}, Band={best_fts_params['band']}, Inner Acc={best_fts_score:.4f}")
            
            # 4. Evaluate on Test Set using Best FTS
            # Extract features for Train (use cache)
            Z_best_train = fgsft.extract_fts_features_from_epoch(
                X_train, fs, best_fts_params['t_indices'], best_fts_params['band'], PTC_PARAMS,
                X_filtered=filtered_cache[best_fts_params['band']]
            )
            
            # Extract features for Test (compute on fly)
            Z_best_test = fgsft.extract_fts_features_from_epoch(
                X_test, fs, best_fts_params['t_indices'], best_fts_params['band'], PTC_PARAMS
            )
            
            # Normalize
            scaler = StandardScaler()
            Z_best_train_scaled = scaler.fit_transform(Z_best_train)
            Z_best_test_scaled = scaler.transform(Z_best_test)
            
            # Train Final Model
            clf_final = LinearSVC(dual='auto', random_state=RANDOM_STATE, max_iter=10000)
            clf_final.fit(Z_best_train_scaled, y_train)
            
            # Predict
            y_test_pred = clf_final.predict(Z_best_test_scaled)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            outer_fold_accs.append(test_acc)
            print(f"    Test Acc: {test_acc:.4f}")
            
            # Record detailed result
            results.append({
                'Subject': subject_id,
                'Fold': fold_idx + 1,
                'Test_Acc': test_acc,
                'Best_Inner_Acc': best_fts_score,
                'Best_Time_Start': best_fts_params['t_window_rel'][0],
                'Best_Time_End': best_fts_params['t_window_rel'][1],
                'Best_Freq_Low': best_fts_params['band'][0],
                'Best_Freq_High': best_fts_params['band'][1]
            })
            
    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv('phase1_results.csv', index=False)
    
    # Summary
    print("\n=== Phase 1 Summary ===")
    summary = df_results.groupby('Subject')['Test_Acc'].agg(['mean', 'std'])
    print(summary)
    print(f"\nOverall Mean Accuracy: {df_results['Test_Acc'].mean():.4f}")

if __name__ == "__main__":
    # Define subjects
    # Assuming standard BCI IV 2a subjects 1-9
    # Or check what load_data supports.
    # The user's environment might have specific subjects.
    # Let's try subject 1 first as a test, or list available.
    # For now, let's assume '1' is available.
    # subjects = ['1'] 
    # If you want to run all, uncomment below:
    subjects = [str(i) for i in range(1, 10)]
    
    run_phase1_analysis(subjects)

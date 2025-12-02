import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

from . import config
from . import io
from . import preprocess
from . import features
from . import fdcc
from . import fusion
from . import ensemble

def evaluate_session(subject_id, session_id=None):
    """
    Run the DSFE pipeline for a single subject/session.
    
    Args:
        subject_id: Subject identifier
        session_id: Not used in current load_data, but kept for interface.
        
    Returns:
        mean_acc: Mean accuracy over folds
        results: Dictionary with detailed results
    """
    print(f"Processing Subject {subject_id}...")
    
    # 1. Load Data
    X_raw, y_raw, fs_raw = io.load_subject_data(subject_id)
    
    # 2. Preprocess
    # Note: We pass fs_raw, but preprocess might resample if we implemented resampling.
    # Current preprocess just epochs and filters.
    X, y = preprocess.preprocess_data(X_raw, y_raw, fs_raw)
    
    print(f"  Data shape: {X.shape}, Labels: {len(y)}")
    
    # 3. Cross-Validation
    cv = StratifiedKFold(n_splits=config.N_FOLDS_EVAL, shuffle=True, random_state=config.RANDOM_STATE)
    
    accuracies = []
    
    fold = 0
    for train_idx, test_idx in cv.split(X, y):
        fold += 1
        print(f"  Fold {fold}/{config.N_FOLDS_EVAL}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # --- Feature Extraction & Band Selection ---
        
        # FTA
        F_fta_train = None
        F_fta_test = None
        band_fta = config.DEFAULT_BAND
        
        if config.USE_FTA:
            if config.USE_FDCC:
                print("    [FDCC] Selecting band for FTA...")
                band_fta = fdcc.fdcc_select_band(X_train, y_train, fs_raw, 'fta')
                print(f"      Best band (FTA): {band_fta}")
            
            F_fta_train = features.compute_fta_features(X_train, fs_raw, band_fta)
            F_fta_test = features.compute_fta_features(X_test, fs_raw, band_fta)
            
        # RG
        F_rg_train = None
        F_rg_test = None
        band_rg = config.DEFAULT_BAND
        
        if config.USE_RG:
            if config.USE_FDCC:
                print("    [FDCC] Selecting band for RG...")
                band_rg = fdcc.fdcc_select_band(X_train, y_train, fs_raw, 'rg')
                print(f"      Best band (RG): {band_rg}")
            
            # RG needs training data mean for tangent space mapping
            F_rg_train, meta = features.compute_rg_features(X_train, fs_raw, band_rg)
            F_rg_test, _ = features.compute_rg_features(X_test, fs_raw, band_rg, training_data=meta)
            
        # --- Fusion ---
        
        feature_sets_train = []
        feature_sets_test = []
        
        # Add basic feature sets if they exist
        if F_fta_train is not None:
            feature_sets_train.append(F_fta_train)
            feature_sets_test.append(F_fta_test)
            
        if F_rg_train is not None:
            feature_sets_train.append(F_rg_train)
            feature_sets_test.append(F_rg_test)
            
        # Add Fused set if enabled and we have both
        if config.USE_RELIEFF and F_fta_train is not None and F_rg_train is not None:
            print("    [ReliefF] Fusing features...")
            F_fused_train, idx_sel = fusion.fuse_features_with_relieff(F_fta_train, F_rg_train, y_train)
            
            # Apply same selection to test
            F_all_test = np.concatenate([F_fta_test, F_rg_test], axis=1)
            F_fused_test = F_all_test[:, idx_sel]
            
            feature_sets_train.append(F_fused_train)
            feature_sets_test.append(F_fused_test)
            
        if not feature_sets_train:
            raise ValueError("No features selected! Enable USE_FTA or USE_RG.")
            
        # --- Classification ---
        
        if config.USE_ENSEMBLE:
            ens = ensemble.DSFEEnsemble()
            classes = np.unique(y) # Assumes all classes present in train
            
            for F_tr in feature_sets_train:
                model = ensemble.FeatureSetModel()
                model.fit(F_tr, y_train)
                ens.add_feature_model(model)
                
            y_pred = ens.predict(feature_sets_test, classes)
            
        else:
            # Fallback: Simple SVM on concatenated features
            print("    [Fallback] Using simple SVM...")
            X_concat_train = np.concatenate(feature_sets_train, axis=1)
            X_concat_test = np.concatenate(feature_sets_test, axis=1)
            
            clf = SVC(kernel='rbf', random_state=config.RANDOM_STATE)
            clf.fit(X_concat_train, y_train)
            y_pred = clf.predict(X_concat_test)
            
        acc = accuracy_score(y_test, y_pred)
        print(f"    Fold {fold} Accuracy: {acc:.4f}")
        accuracies.append(acc)
        
    mean_acc = np.mean(accuracies)
    print(f"Subject {subject_id} Mean Accuracy: {mean_acc:.4f}")
    
    return mean_acc, {'accuracies': accuracies}

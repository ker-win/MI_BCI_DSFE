import numpy as np
from sklearn.neighbors import NearestNeighbors
from . import config

class SimpleReliefF:
    """
    A simplified implementation of ReliefF algorithm.
    """
    def __init__(self, n_neighbors=10, n_features_to_select=10):
        self.n_neighbors = n_neighbors
        self.n_features_to_select = n_features_to_select
        self.feature_importances_ = None
        self.top_features_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        
        # Normalize features to [0, 1] for distance calculation
        # (Important for ReliefF)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        range_X = X_max - X_min
        range_X[range_X == 0] = 1 # Avoid division by zero
        X_norm = (X - X_min) / range_X
        
        classes = np.unique(y)
        
        # Pre-compute nearest neighbors for efficiency
        # We need neighbors from same class (Hits) and different classes (Misses)
        
        # Separate data by class
        X_by_class = {c: X_norm[y == c] for c in classes}
        
        # Iterations: Paper says "randomly select a sample S"
        # We can iterate over all samples or a subset. 
        # config.N_ITER_RELIEFF controls this.
        
        n_iter = config.N_ITER_RELIEFF
        if n_iter > n_samples:
            n_iter = n_samples
            indices = np.arange(n_samples)
        else:
            indices = np.random.choice(n_samples, n_iter, replace=False)
            
        for idx in indices:
            S = X_norm[idx]
            target_class = y[idx]
            
            # Find Hits (k nearest neighbors from same class)
            # We must exclude S itself
            X_hits = X_by_class[target_class]
            if len(X_hits) > 1:
                # Calculate distances to S
                dists = np.linalg.norm(X_hits - S, axis=1)
                # Sort and take k (excluding self which is dist 0)
                # Note: S is in X_hits
                nearest_indices = np.argsort(dists)
                # Skip 0 (self)
                hits_indices = nearest_indices[1:self.n_neighbors+1]
                # If not enough neighbors, take what we have
                if len(hits_indices) == 0:
                     hits = []
                else:
                     hits = X_hits[hits_indices]
            else:
                hits = []
                
            # Find Misses (k nearest neighbors from EACH other class)
            misses = {}
            for c in classes:
                if c == target_class:
                    continue
                X_miss = X_by_class[c]
                if len(X_miss) == 0:
                    continue
                dists = np.linalg.norm(X_miss - S, axis=1)
                nearest_indices = np.argsort(dists)[:self.n_neighbors]
                misses[c] = X_miss[nearest_indices]
                
            # Update weights
            # W[A] = W[A] - diff(A, S, H)/m + sum(P(C)*diff(A, S, M(C)))/m
            # Simplified: average diffs
            
            for i in range(n_features):
                # Diff with Hits
                diff_hits = 0
                if len(hits) > 0:
                    diff_hits = np.sum(np.abs(S[i] - hits[:, i])) / (len(hits) * n_iter)
                
                # Diff with Misses
                diff_misses = 0
                total_miss_prob = 0
                
                for c, M_list in misses.items():
                    if len(M_list) > 0:
                        # P(C) can be estimated by class frequency or 1/(n_classes-1)
                        # Paper doesn't specify, usually P(C) is prior prob.
                        prob_c = len(X_by_class[c]) / n_samples
                        diff_c = np.sum(np.abs(S[i] - M_list[:, i])) / (len(M_list) * n_iter)
                        diff_misses += prob_c * diff_c
                        total_miss_prob += prob_c
                
                # Normalize miss term if needed, but standard formula uses P(C)
                # If we sum P(C) over all C!=target, it is 1-P(target).
                # Standard ReliefF: sum_C!=target [ P(C)/(1-P(target)) * diff(A,S,M(C)) ]
                
                p_target = len(X_hits) / n_samples
                if 1 - p_target > 0:
                     diff_misses /= (1 - p_target)
                
                self.feature_importances_[i] += diff_misses - diff_hits

        # Select top features
        self.top_features_ = np.argsort(self.feature_importances_)[::-1][:self.n_features_to_select]
        return self

    def transform(self, X):
        return X[:, self.top_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def fuse_features_with_relieff(F_fta, F_rg, y):
    """
    Concatenate features and apply ReliefF.
    
    Args:
        F_fta: (n_trials, n_features_fta)
        F_rg: (n_trials, n_features_rg)
        y: (n_trials,)
        
    Returns:
        F_fused: (n_trials, n_selected)
        selected_indices: Indices of selected features in the concatenated array
    """
    F_all = np.concatenate([F_fta, F_rg], axis=1)
    
    n_features = F_all.shape[1]
    n_keep = int(n_features * config.RELIEFF_KEEP_RATIO)
    
    # Try to use skrebate if available, else use custom
    try:
        from skrebate import ReliefF
        relieff = ReliefF(n_features_to_select=n_keep, n_neighbors=config.N_NEIGHBORS_RELIEFF)
        relieff.fit(F_all, y)
        # skrebate doesn't have a simple attribute for indices sometimes, 
        # but transform works. To get indices:
        # It usually stores feature_importances_
        weights = relieff.feature_importances_
        selected_indices = np.argsort(weights)[::-1][:n_keep]
        F_fused = F_all[:, selected_indices]
        
    except ImportError:
        # Use our simple implementation
        relieff = SimpleReliefF(n_neighbors=config.N_NEIGHBORS_RELIEFF, n_features_to_select=n_keep)
        relieff.fit(F_all, y)
        selected_indices = relieff.top_features_
        F_fused = relieff.transform(F_all)
        
    return F_fused, selected_indices

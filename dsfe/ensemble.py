import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from . import config

class FeatureSetModel:
    """
    A model trained on a specific feature set (e.g., FTA only, RG only, or Fused).
    Contains 3 classifiers: SVM, RF, NB.
    """
    def __init__(self):
        self.svm = SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE)
        self.rf = RandomForestClassifier(n_estimators=config.N_TREES_RF, random_state=config.RANDOM_STATE)
        self.nb = GaussianNB()
        
    def fit(self, X, y):
        self.svm.fit(X, y)
        self.rf.fit(X, y)
        self.nb.fit(X, y)
        
    def predict(self, X):
        """
        Returns predictions from all 3 classifiers.
        Returns:
            dict: {'svm': pred, 'rf': pred, 'nb': pred}
        """
        return {
            'svm': self.svm.predict(X),
            'rf': self.rf.predict(X),
            'nb': self.nb.predict(X)
        }

class DSFEEnsemble:
    """
    The final ensemble that aggregates predictions from multiple FeatureSetModels.
    """
    def __init__(self):
        self.feature_models = []
        
    def add_feature_model(self, model):
        self.feature_models.append(model)
        
    def predict(self, X_list, classes):
        """
        Predict using weighted voting.
        
        Args:
            X_list: List of feature matrices, one for each FeatureSetModel.
            classes: List of unique class labels (e.g. [0, 1, 2, 3])
            
        Returns:
            y_pred: (n_samples,)
        """
        n_samples = X_list[0].shape[0]
        n_classes = len(classes)
        
        # Score matrix: (n_samples, n_classes)
        scores = np.zeros((n_samples, n_classes))
        
        # Map class label to index
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for X, f_model in zip(X_list, self.feature_models):
            preds = f_model.predict(X)
            
            for name, y_hat in preds.items():
                if name == 'svm':
                    w = config.SVM_WEIGHT
                elif name == 'rf':
                    w = config.RF_WEIGHT
                elif name == 'nb':
                    w = config.NB_WEIGHT
                else:
                    continue
                    
                # Add weights
                for i in range(n_samples):
                    pred_class = y_hat[i]
                    if pred_class in class_to_idx:
                        idx = class_to_idx[pred_class]
                        scores[i, idx] += w
                        
        # Argmax to get final prediction
        final_indices = np.argmax(scores, axis=1)
        y_final = np.array([classes[i] for i in final_indices])
        
        return y_final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

class AnomalyDetectionSuite:
    """
    Implements both Unsupervised (K-Means distance, Gaussian Mixture Model density)
    and Supervised (SMOTE + SVM / Random Forest) anomaly detection models.
    """
    def __init__(self, X_scaled: np.ndarray, y_true: np.ndarray, contamination=0.025):
        self.X = X_scaled
        self.y = y_true
        self.contamination = contamination
        self.results = {}
        
    def run_kmeans_detector(self, n_clusters=4):
        """Flags points furthest from their assigned K-Means cluster centroid."""
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(self.X)
        
        # Compute Euclidean distance to nearest cluster center
        distances = np.min(kmeans.transform(self.X), axis=1)
        threshold = np.percentile(distances, 100 * (1 - self.contamination))
        preds = (distances >= threshold).astype(int)
        
        self._log_metrics("Unsupervised K-Means (Distance)", self.y, preds, distances)
        return preds, distances

    def run_gmm_detector(self, n_components=4):
        """Flags points in low-density regions using Gaussian Mixture Model log-likelihood."""
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(self.X)
        
        # Negative log-likelihood score (higher = more anomalous)
        anomaly_scores = -gmm.score_samples(self.X)
        threshold = np.percentile(anomaly_scores, 100 * (1 - self.contamination))
        preds = (anomaly_scores >= threshold).astype(int)
        
        self._log_metrics("Unsupervised GMM (Log-Likelihood)", self.y, preds, anomaly_scores)
        return preds, anomaly_scores

    def run_supervised_classifiers(self):
        """
        Balances severe class imbalance using SMOTE and compares 
        RBF-Kernel SVM vs. Random Forest on a held-out test split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.30, stratify=self.y, random_state=42
        )
        
        # Handle extreme anomaly class imbalance on training set using SMOTE
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        # 1. Support Vector Machine (RBF Kernel)
        svm_model = SVC(kernel='rbf', C=1.5, probability=True, random_state=42)
        svm_model.fit(X_train_bal, y_train_bal)
        svm_preds = svm_model.predict(X_test)
        svm_probs = svm_model.predict_proba(X_test)[:, 1]
        self._log_metrics("Supervised SVM (RBF + SMOTE)", y_test, svm_preds, svm_probs)
        
        # 2. Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
        rf_model.fit(X_train_bal, y_train_bal)
        rf_preds = rf_model.predict(X_test)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        self._log_metrics("Supervised Random Forest (SMOTE)", y_test, rf_preds, rf_probs)
        
        return pd.DataFrame(self.results).T

    def _log_metrics(self, model_name, y_true, y_pred, scores):
        self.results[model_name] = {
            'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
            'F1_Score': round(f1_score(y_true, y_pred, zero_division=0), 4),
            'ROC_AUC': round(roc_auc_score(y_true, scores), 4)
        }
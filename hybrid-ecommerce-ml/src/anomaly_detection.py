import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class DBSCANAnomalyGenerator:
    """DBSCAN Session Anomaly Indicator Generator using core sample distance thresholds for out-of-sample inference."""
    def __init__(self, eps=1.8, min_samples=15):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.dbscan = None
        self.nn_core = None

    def fit(self, X_num):
        X_scaled = self.scaler.fit_transform(X_num)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.dbscan.fit_predict(X_scaled)
        
        # Fit Nearest Neighbors on non-outlier training core points for test-time scoring
        core_mask = labels != -1
        if np.sum(core_mask) > 0:
            self.nn_core = NearestNeighbors(n_neighbors=1).fit(X_scaled[core_mask])
        else:
            self.nn_core = NearestNeighbors(n_neighbors=1).fit(X_scaled)
        return self

    def transform(self, X_num):
        X_scaled = self.scaler.transform(X_num)
        if self.nn_core is None:
            return np.zeros(len(X_num), dtype=int)
            
        distances, _ = self.nn_core.kneighbors(X_scaled)
        # Session is an anomaly if its distance to nearest training cluster exceeds eps
        is_outlier = (distances.flatten() > self.eps).astype(int)
        return is_outlier

    def fit_transform(self, X_num):
        self.fit(X_num)
        X_scaled = self.scaler.transform(X_num)
        return (self.dbscan.labels_ == -1).astype(int)
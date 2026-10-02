import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeansFeatureGenerator:
    """K-Means behavioral cluster feature generator trained strictly on numerical engagement features."""
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_names = {}

    def fit(self, X_num):
        X_scaled = self.scaler.fit_transform(X_num)
        self.kmeans.fit(X_scaled)
        
        # Analyze cluster centers to assign evidence-based behavioral titles
        centers = pd.DataFrame(self.scaler.inverse_transform(self.kmeans.cluster_centers_), columns=X_num.columns)
        for c_id in range(self.n_clusters):
            row = centers.iloc[c_id]
            if row['PageValues'] > 15:
                self.cluster_names[c_id] = "High-Intent Buyers"
            elif row['ProductRelated_Duration'] > 1200:
                self.cluster_names[c_id] = "Deep Product Researchers"
            elif row['BounceRates'] > 0.08 or row['ExitRates'] > 0.08:
                self.cluster_names[c_id] = "Low Engagement / Bouncers"
            else:
                self.cluster_names[c_id] = "General Browsers"
        return self

    def transform(self, X_num):
        X_scaled = self.scaler.transform(X_num)
        return self.kmeans.predict(X_scaled)

    def fit_transform(self, X_num):
        self.fit(X_num)
        return self.transform(X_num)
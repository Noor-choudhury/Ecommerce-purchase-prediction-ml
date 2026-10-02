import numpy as np
import pandas as pd
from src.preprocessing import NUMERICAL_COLS

class UnsupervisedSupervisedEcommercePipeline:
    """Serialization wrapper encapsulating preprocessing, clustering, anomaly detection, TF-IDF mapping, and Random Forest prediction."""
    def __init__(self, preprocessor, kmeans_gen, dbscan_gen, tfidf_gen, rf_model, feature_names):
        self.preprocessor = preprocessor
        self.kmeans_gen = kmeans_gen
        self.dbscan_gen = dbscan_gen
        self.tfidf_gen = tfidf_gen
        self.rf_model = rf_model
        self.feature_names = feature_names

    def predict_session(self, input_data):
        """Processes raw single session input (dict or DataFrame) and returns purchase probability and predictions."""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # 1. Base Preprocessing
        X_base = self.preprocessor.transform(input_df)
        
        # 2. K-Means Cluster Assignment
        X_num = input_df[NUMERICAL_COLS]
        cluster_id = int(self.kmeans_gen.transform(X_num)[0])
        cluster_name = self.kmeans_gen.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        
        # 3. DBSCAN Anomaly Indicator
        is_outlier = int(self.dbscan_gen.transform(X_num)[0])
        
        # 4. TF-IDF Cosine Similarity
        sim_scores, matched_df = self.tfidf_gen.transform(input_df)
        sim_score = float(sim_scores[0])
        matched_product = matched_df.iloc[0].to_dict()
        
        # 5. Combined Feature Vector Assembly
        X_combined = np.column_stack([X_base, cluster_id, is_outlier, sim_score])
        
        # 6. Final Classification Output
        prob = float(self.rf_model.predict_proba(X_combined)[0, 1])
        prediction = int(prob >= 0.5)
        
        return {
            "purchase_probability": prob,
            "prediction": prediction,
            "prediction_label": "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "is_outlier": is_outlier,
            "anomaly_status": "Session Anomaly Detected" if is_outlier == 1 else "Normal Behavior",
            "similarity_score": sim_score,
            "matched_product": matched_product
        }
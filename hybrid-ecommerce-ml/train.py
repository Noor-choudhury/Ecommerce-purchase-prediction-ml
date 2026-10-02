import os
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import find_dataset_path, load_and_clean_data, split_data, create_preprocessor, NUMERICAL_COLS
from src.clustering import KMeansFeatureGenerator
from src.anomaly_detection import DBSCANAnomalyGenerator
from src.text_features import TFIDFSimilarityFeatureGenerator
from src.model_training import get_prediction_model
from src.evaluation import evaluate_model_performance, plot_confusion_matrix_fig, plot_roc_curve_fig, plot_feature_importance_fig
from src.pipeline import UnsupervisedSupervisedEcommercePipeline

def main():
    print("=== Unsupervised+Supervised Prediction Model Training Pipeline ===")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Dataset Loading
    data_path = find_dataset_path()
    print(f"Loading dataset from: {data_path}")
    df = load_and_clean_data(data_path)
    print(f"Cleaned dataset shape: {df.shape}")
    
    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    
    # 3. Preprocessing Setup
    preprocessor = create_preprocessor()
    X_train_base = preprocessor.fit_transform(X_train)
    X_test_base = preprocessor.transform(X_test)
    
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
    base_feature_names = list(NUMERICAL_COLS) + list(cat_feature_names)
    
    # 4. K-Means Feature Generation
    print("Fitting K-Means Clustering...")
    kmeans_gen = KMeansFeatureGenerator(n_clusters=4, random_state=42).fit(X_train[NUMERICAL_COLS])
    train_clusters = kmeans_gen.transform(X_train[NUMERICAL_COLS])
    test_clusters = kmeans_gen.transform(X_test[NUMERICAL_COLS])
    
    # 5. DBSCAN Anomaly Detection
    print("Fitting DBSCAN Anomaly Detector...")
    dbscan_gen = DBSCANAnomalyGenerator(eps=1.8, min_samples=15).fit(X_train[NUMERICAL_COLS])
    train_outliers = dbscan_gen.fit_transform(X_train[NUMERICAL_COLS])
    test_outliers = dbscan_gen.transform(X_test[NUMERICAL_COLS])
    
    # 6. TF-IDF Cosine Similarity Engine
    print("Fitting TF-IDF Similarity Engine...")
    tfidf_gen = TFIDFSimilarityFeatureGenerator().fit()
    train_sims, _ = tfidf_gen.transform(X_train)
    test_sims, _ = tfidf_gen.transform(X_test)
    
    # 7. Construct Feature Matrix
    X_train_combined = np.column_stack([X_train_base, train_clusters, train_outliers, train_sims])
    X_test_combined = np.column_stack([X_test_base, test_clusters, test_outliers, test_sims])
    combined_feature_names = base_feature_names + ['cluster_id', 'is_outlier', 'similarity_score']
    
    # 8. Model Training & Evaluation
    final_rf = get_prediction_model(random_state=42)
    final_rf.fit(X_train_combined, y_train)
    
    y_pred_final = final_rf.predict(X_test_combined)
    y_prob_final = final_rf.predict_proba(X_test_combined)[:, 1]
    
    metrics = evaluate_model_performance(y_test, y_pred_final, y_prob_final)
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    # Visualizations
    plot_confusion_matrix_fig(y_test, y_pred_final, save_path="outputs/confusion_matrix.png")
    plot_roc_curve_fig(y_test, y_prob_final, save_path="outputs/roc_curve.png")
    plot_feature_importance_fig(final_rf, combined_feature_names, save_path="outputs/feature_importance.png")
    
    # 9. Pipeline Serialization
    pipeline = UnsupervisedSupervisedEcommercePipeline(
        preprocessor=preprocessor,
        kmeans_gen=kmeans_gen,
        dbscan_gen=dbscan_gen,
        tfidf_gen=tfidf_gen,
        rf_model=final_rf,
        feature_names=combined_feature_names
    )
    
    joblib.dump(preprocessor, "models/preprocessing.pkl")
    joblib.dump(kmeans_gen, "models/kmeans.pkl")
    joblib.dump(dbscan_gen, "models/dbscan.pkl")
    joblib.dump(tfidf_gen, "models/tfidf.pkl")
    joblib.dump(final_rf, "models/random_forest.pkl")
    joblib.dump(pipeline, "models/ecommerce_pipeline.pkl")
    
    print("\nAll artifacts saved successfully to models/ and outputs/.")

if __name__ == "__main__":
    main()